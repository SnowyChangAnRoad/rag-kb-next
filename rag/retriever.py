"""检索层 | Day9：向量召回（bi-encoder） + 精排（cross-encoder） + 冗余去重。

两阶段检索（Day8 已口述，今天固化成代码）：
    召回  bi-encoder     快 / 求全：doc 向量可离线预计算 + ANN → 候选池 top RECALL_K
    精排  cross-encoder  准 / 求准：query+doc 拼成一个序列做 full attention → 降序重排
    去重  （今天新增）    10% overlap 让相邻块高度相似，精排会把同源几块一起推上去，
                        把另一侧的跨文档证据挤出 prompt（Day6「同段重复 4 条」、
                        Day8 Q7「top3 全被 RAG知识.pdf 占掉」都是这个病）
"""
import re

import chromadb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    DB_PATH, DEFAULT_EMBED, DEFAULT_VARIANT, RECALL_K, RERANKER_MODEL_PATH,
    resolve_variant,
)
from .embedding import encode_query, get_embedder


# ------------------------------------------------------------
# 连接 + 召回
# ------------------------------------------------------------
def get_client() -> chromadb.ClientAPI:
    # 持久化客户端
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(client=None, model_key: str = DEFAULT_EMBED,
                   variant: str = DEFAULT_VARIANT):
    """已给：取某个嵌入模型某一档 collection（cosine 空间），不存在则创建空库。

    variant 是第二个轴，它同时决定两件事 —— 库名 与 向量维度（都挂在
    config.EMBED_MODELS[model_key]["variants"][variant] 下）：
      · bge：raw（原文库）/ ctx（语境前缀库，卡5）；
      · qwen3：raw(1024) / 512 / 256（MRL 截断，卡6）。
    默认 raw，所以 W1/W2 已有的调用（api/cli/eval/fusion）一行都不用改。

    ⚠️ 一条硬约束：variant 只切档，【不切模型】。库名与嵌入模型同挂在一条记录里，
    就是因为"建某档时换了模型、查询时还用旧模型"这类错只能靠结构拦住 ——
    collection 的维度在第一次写入时就被 HNSW 定死：
      · 维度不同      → HNSW 直接报错（幸运，当场拦住）；
      · 维度相同但换模型 / 换了截断位置 → 两套不可比的向量混进同一空间，距离失去意义
        且【不报错】（危险）。
    embedding_function 只服务 upsert（建库）侧的自动编码，也必须按【本档维度】截断；
    查询侧的编码见 retrieve()。
    """
    client = client or get_client()
    v = resolve_variant(model_key, variant)
    return client.get_or_create_collection(
        name=v["collection"],
        embedding_function=get_embedder(model_key, variant),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(collection, question: str, k: int = RECALL_K,
             model_key: str = DEFAULT_EMBED,
             variant: str = DEFAULT_VARIANT) -> list[dict]:
    # 向量召回，返回长度 k 的 hits，每项含 chunk_id / distance / text / 元数据。
    # 查询向量在这里【显式】编码、用 query_embeddings 传，不再走 query_texts，两个理由：
    #   ① instruction-aware 模型的指令只能加在查询侧，而 Chroma 的 embedding_function
    #      被 add/query 共用、分不出该不该加指令（W2 卡3 的根因）；
    #   ② 卡6 的 MRL 截断必须在【查询向量】上也做一遍 —— 两侧维度不一致就没法比距离。
    # 注意 query 会先过嵌入模型（走 GPU），所以这一步的耗时里既有模型前向也有 HNSW 检索。
    # ⚠️ variant 必须与 collection 的档一致（它决定查询向量截到多少维）。
    qvec = encode_query(model_key, [question], variant)
    output = collection.query(query_embeddings=qvec, n_results=k)
    hits = []
    for dist, doc, meta, cid in zip(
        output["distances"][0], output["documents"][0],
        output["metadatas"][0], output["ids"][0],
    ):
        hits.append({"chunk_id": cid, "distance": dist, "text": doc, **meta})
    return hits


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """已给：'RAG知识_p076_c001' -> ('RAG知识_p076', 1)；不匹配时返回 (原串, -1)。

    去重要用它：同一 doc_id 内序号相差 ≤1 的两块，就是 overlap 造就的「相邻冗余块」。
    """
    m = re.match(r"^(.*)_c(\d+)$", chunk_id)
    return (m.group(1), int(m.group(2))) if m else (chunk_id, -1)


# ------------------------------------------------------------
# 精排（Day8 搬运，只有一处行为变化：不再内部截断）
# ------------------------------------------------------------
class BGEReranker:
    def __init__(self, model_path: str = RERANKER_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # bge-reranker-base 结构 = BERT + 一层输出 1 个值的分类头
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda").eval()

    def score(self, question: str, hits: list[dict]) -> list[float]:
        """已给：给 (query, doc) 对打分，返回 0~1 的分数列表（sigmoid 后的概率）。"""
        pairs = [[question, h["text"]] for h in hits]
        tokens = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            logits = self.model(**tokens).logits
        return torch.sigmoid(logits.squeeze(-1).float()).cpu().tolist()

    def rerank(self, question: str, hits: list[dict]) -> list[dict]:
        """已给：打分 → 写回 rerank_score → 按精排分降序返回【全量】，不截断。

        与 Day8 的一处差别：08 在方法内部就切了 top3。今天「去重」必须插在截断之前，
        所以截断上移到 pipeline —— 顺序错了，去重腾出的槽位就白费
        （见 pipeline._retrieve 的 docstring「顺序不能错：必须精排 → 去重 → 截断」）。
        注意 hits 里是 dict（可变对象），rerank_score 会写回原列表，调用方可原地对照。
        """
        scores = self.score(question, hits)
        for h, s in zip(hits, scores):
            h["rerank_score"] = s
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

# 去重 —— 同一文档内的「相邻块」只保留一条
def dedupe_adjacent(hits: list[dict]) -> list[dict]:
    """
    背景：chunk_size=300 / overlap=30，相邻块共享约 30 字，embedding 与精排分都极接近；
    精排是逐块独立打分的，没有「互相排斥」机制，于是同源一块接一块挤进前三。

    「相邻」判定（今天定的口径，通关时要能说清）：
      - 用 parse_chunk_id 拆出 (doc_id, seq)
      - doc_id 相同 且 |seq 差| ≤ 1 → 视为同一段内容的重叠冗余
      - 输入已按精排分降序，所以先遇到的就是这一簇里最高分的那块

    要求：
      - 返回【新列表】，顺序保持输入的降序（别原地改 hits，调用方还要对照）

    注意一个容易漏的点：本规则只治「同页相邻」（PDF 的 doc_id 是「文档_页码」）。
    跨页的 p076_c002 与 p077_c001 也是相邻冗余，但 doc_id 不同 → 今天先不治，留实验记录。
    """
    kept, keys = [], []
    for h in hits:
        doc, seq = parse_chunk_id(h["chunk_id"])
        if any(doc == d and abs(seq - s) <= 1 for d, s in keys):
            continue
        kept.append(h)
        keys.append((doc, seq))
    return kept

# 同源配额 + 截断
def select_diverse(hits: list[dict], k: int, per_source_cap: int) -> list[dict]:
    """
    为什么去重治不了它（今天卡 1 的 Q2 答案）：
        cross-encoder 对每个 (query, doc_i) 独立打分，块与块之间没有交互项，
        同源相似块会一起被推高。dedupe_adjacent 只处理「同页且序号差≤1」的
        重叠冗余；今天实测 Q1 的 top3 = p15, p16_c001, p16_c003（序号差 2）
        照样一起入榜 → 槽位浪费，覆盖率只有 36%。
        配额换的是另一个维度：不再问「这两块像不像」，而是问「这个来源占了几条」。

    输入约定：hits 已按精排分降序（rerank 的输出）——先遇到的就是每个 source 的最高分块。
    输出：长度尽量为 k 的列表，仍保持降序。

    两轮选择（第二轮回填是不许跳的关键）：
        第一轮  遍历 hits（**收满 k 条就 break**，不然第一轮自己就超员了）：
                某 source 已取满 per_source_cap 条 → 暂存进 overflow；
                否则收进 picked，并把该 source 计数 +1。
        第二轮  若 picked 不足 k 条（只有「遍历完都没凑满」才可能发生，例如候选池
                只有一个 source），按原顺序把 overflow 里的块补回来 ——
                配额的作用是「挤掉冗余」，不是「少给资料」，凑不满 k 会白丢证据。

    自检（纯本地，不花钱）：
        ① hits = [{"source":"a"},{"source":"a"},{"source":"b"},{"source":"a"},{"source":"b"}]
           （已按精排降序）→ select_diverse(hits, 3, 2) 应返回 3 条：a, a, b
             a 占满 2 条后，第 4 条 a 被压掉；返回的是前 3 条，不是 4 条。
        ② hits = [{"source":"a"}] * 4 → select_diverse(hits, 3, 2) 长度必须还是 3
             （只回填凑数，宁可给同源冗余，也不能只给 2 条）
    """
    picked:list[dict] = []
    overflow:list[dict] = []
    used: dict[str, int] = {}          # ← source 的记录表

    for h in hits:
        if len(picked) >= k:
            break
        src = h["source"]
        if used.get(src,0) >= per_source_cap:     # get(src,0) 0是找不到键时的默认返回值
            overflow.append(h)
        else:
            picked.append(h)
            used[src] = used.get(src,0) + 1

    if len(picked) < k:
        for h in overflow:
            if len(picked) >= k:
                break
            picked.append(h)

    return picked
