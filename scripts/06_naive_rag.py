"""
Day 6 | 06_naive_rag.py
目标：最小 RAG 闭环——全量入库 chunks.jsonl → 检索 k=5 → 拼 prompt 调 DeepSeek 生成。
第一次把「检索」和「生成」连起来：检索不到好资料，生成再强也是幻觉。

用法（在项目根 rag-kb 下）：
    python scripts/06_naive_rag.py build   # 全量入库：chunks.jsonl → day6_kb（幂等，可重跑）
    python scripts/06_naive_rag.py         # 交互问答：输入问题 → 打印命中片段 → 打印回答，q 退出

链路（通关要能口述）：
    问题 → BGE 向量化 → Chroma 查 top-k（cosine 距离） → 拼进 prompt → DeepSeek 回答

设计决策：
1. 新建 day6_kb，不碰 day2_kb：玩具语料保留作对照，真实知识库独立成库。
2. build 用 upsert 不用 add：重跑不报错、不重复（Day2 已学，今天复习）。
3. 分批入库（每批 64 条）：BGE 一次编码 531 条会把 padding 全拉到最长，分批更稳，
   且能打印进度。
4. metadata 坑：Chroma 的 metadata 值只接受 str/int/float/bool——网页块 page=None、
   PDF 块 url=None 会直接报错，入库前必须把值为 None 的键整个删掉。
5. Day6 是朴素版 prompt：只拼资料+问题，不搞引用编号和严格拒答（那是 Day7 的事），
   但 system 里先埋一句"资料不足就说没有"，防止明显编造。

硬边界：k 固定 5；非流式输出（流式 Day12 才做）；不做 rerank（Day8）。
"""
import os
import sys
import json
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from common_embedding import BGEEmbeddingFunction

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL = ROOT / "data" / "processed" / "chunks.jsonl"
DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "day6_kb"

load_dotenv(str(ROOT / ".env"))   # 显式指定路径，不依赖当前工作目录

K = 5                # 检索条数（计划要求 k=5）
BATCH_SIZE = 64      # 入库分批大小

# ------------------------------------------------------------
# Prompt 模板（Day6 朴素版，Day7 会升级成引用+拒答）
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一个基于个人知识库的问答助手，只能依据用户提供的资料回答问题。"
    "如果资料中没有足够信息，就回答\"资料中没有相关内容\"。用中文，简洁准确。"
)
USER_TEMPLATE = "资料：\n{context}\n\n问题：{question}"


# ------------------------------------------------------------
# 辅助函数（已给全）
# ------------------------------------------------------------
def safe_preview(text: str, n: int = 40) -> str:
    """控制台只打印短预览，避免特殊 Unicode 触发 cp936 编码错误（Day3 教训）。"""
    return text.replace("\n", " ")[:n]


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(client: chromadb.ClientAPI):
    """query 时 Chroma 需要同一个嵌入函数把问题向量化（Day2 已学）。"""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def get_llm() -> OpenAI:
    """DeepSeek 走 OpenAI 兼容 SDK：只换 base_url，代码与调 GPT 完全一样。"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def show_hits(hits: list[dict]) -> None:
    """打印命中片段：排名 / 距离 / 来源（页码或链接）/ 前 40 字预览。"""
    for rank, h in enumerate(hits, 1):
        if h.get("page") is not None:
            where = f"第{h['page']}页"
        else:
            where = h.get("url") or "?"
        print(f"  #{rank} d={h['distance']:.4f}  {h['source']} {where}")
        print(f"      {safe_preview(h['text'])}")


# ============================================================
# TODO 1（热身）：读 chunks.jsonl，约 5 行
#   参考自己 Day5 写的 load_documents：逐行 json.loads，跳过空行
# ============================================================
def load_chunks() -> list[dict]:
    docs = []
    with open(CHUNKS_JSONL,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# ============================================================
# TODO 2（核心）：全量入库，约 15 行
#   2a: clean_meta —— 用字典推导式把值为 None 的键删掉
#   2b: 按 BATCH_SIZE 分批 upsert（切批可用 range(0, len, BATCH_SIZE) 切片）
#       每批打印进度（第几批 / 已入库多少条），最后 collection.count() 验证
#   注意 documents 是每块的 text，metadatas 每条带 chunk_id/doc_id/source/page/url/title
# ============================================================
def cmd_build() -> None:
    chunks = load_chunks()
    print(f"读入 {len(chunks)} 块，目标库：{COLLECTION_NAME}（cosine）")

    def clean_meta(meta: dict) -> dict:
        # TODO 2a：删掉值为 None 的键后返回
        return {k:v for k, v in meta.items() if v is not None}   # 字典推导式，删除值为None的键值对

    collection = get_collection(get_client())
    # TODO 2b 答案：先切 batch，再从同一个 batch 推导三个列表，避免写三遍 range 切片
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # 每条块的 metadata（text 已作为 document，不重复放）；clean_meta 删掉 None 键
        metadatas = [
            clean_meta({
                "chunk_id": c["chunk_id"],      # id留在metadatas中在检索时可用，并不冗余
                "doc_id": c["doc_id"],
                "source": c["source"],
                "page": c["page"],      # 网页块为 None → 被 clean_meta 删掉
                "url": c["url"],        # PDF 块为 None → 被 clean_meta 删掉
                "title": c["title"],
            })
            for c in batch
        ]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=metadatas,
        )
        print(f"已入库 {min(start + BATCH_SIZE, total)}/{total}")
    print("=" * 60)
    print(f"入库完成，collection.count() = {collection.count()}")


# ============================================================
# TODO 3：检索，约 8 行
#   collection.query(query_texts=[question], n_results=k)
#   返回的是"批"结构（每个列表外面套一层），取 [0] 后 zip 四个字段，
#   整理成 [{"chunk_id", "distance", "text", ...metadata}, ...] 返回。
#   show_hits 会用到 distance / text / source / page / url 这些键。
# ============================================================
def retrieve(collection, question: str, k: int = K) -> list[dict]:
    # TODO 3
    output = collection.query(query_texts=[question], n_results=k)
    result = []
    for rank, (dist, doc, meta, id) in enumerate(
        zip(output["distances"][0], output["documents"][0], output["metadatas"][0], output["ids"][0]),
        start=1
    ):
        result.append({"chunk_id":id, "distance":dist, "text":doc, **meta})
    return result


# ============================================================
# TODO 4：拼 context，约 8 行
#   每块格式（块之间空一行）：
#       【资料n】{source} 第x页 或 链接
#       {text}
#   最后套 USER_TEMPLATE 返回完整 user 消息
# ============================================================
def build_prompt(question: str, hits: list[dict]) -> str:
    # TODO 4 答案：逐块拼"来源行 + 正文"，块间空一行，最后套模板（context 必须是字符串）
    blocks = []
    for i, hit in enumerate(hits, 1):
        if hit.get("page") is not None:                 # PDF 块：显示书名 + 页码
            where = f"第{hit['page']}页"
        else:                                            # 网页块：显示 url
            where = hit.get("url") or "网络资料"
        blocks.append(f"【资料{i}】来源：{hit['source']} {where}\n{hit['text']}")
    context = "\n\n".join(blocks)
    return USER_TEMPLATE.format(context=context, question=question)


# ============================================================
# TODO 5（核心）：单次问答，约 12 行
#   1) retrieve 拿 k 条命中
#   2) 打印"检索结果"段（直接调 show_hits）
#   3) build_prompt 拼 user 消息，messages=[{"role": "system", ...}, {"role": "user", ...}]
#   4) llm.chat.completions.create(model="deepseek-v4-flash", messages=messages,
#      temperature=0.7, extra_body={"thinking": {"type": "disabled"}})
#   5) 打印"回答"段：resp.choices[0].message.content
#   6) 打印 token 用量：resp.usage.prompt_tokens / completion_tokens
# ============================================================
def ask_once(collection, llm: OpenAI, question: str) -> None:
    # TODO 5
    hits = retrieve(collection, question, K)
    print("--- 检索结果 ---")
    show_hits(hits)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    # 模型名以官方文档为准：deepseek-chat 已于 2026/07/24 弃用，现名 deepseek-v4-flash
    # v4-flash 默认开思考模式（temperature 不生效），Day6 用 extra_body 关掉
    resp = llm.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.7,
        extra_body={"thinking": {"type": "disabled"}},
    )
    print("--- 回答 ---")
    print(resp.choices[0].message.content)
    print(f"Token 用量：输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "build":
        cmd_build()
        return
    if len(sys.argv) == 1:
        collection = get_collection(get_client())
        llm = get_llm()
        print(f"已加载 {COLLECTION_NAME}（{collection.count()} 条）。输入问题开始，q 退出。")
        while True:
            try:
                q = input("\n你的问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in {"q", "quit", "exit"}:
                break
            ask_once(collection, llm, q)
        return
    print("用法：python scripts/06_naive_rag.py [build]")
    sys.exit(1)


if __name__ == "__main__":
    main()

# ============================================================
# 观察题（跑通后把答案写成注释）：
# O1: build 连跑两次，count 是 531 还是 1062？如果把 upsert 换成 add 会发生什么？
#     531。如果吧upsert换成add则会报错。
# O2: 看你问 5 个问题时打印的 distance：普遍比 Day2 玩具库大还是小？为什么真实文档"语义命中"比玩具语料难？
#     普遍更小。真实文档内容更多，命中更难。
# O3: 同一个问题连问两遍，回答逐字相同吗？temperature=0 时呢？
#     不同。temperture=0时则完全相同。
# 通关 5 问（都应基于资料回答；跑完截图留档）：
#   1. 这份资料主要讲什么？
#   2. RAG 的基本流程是什么？它靠什么减少大模型的幻觉？
#   3. chunk_size 设得太大或太小分别有什么问题？
#   4. 为什么相邻块之间要加 overlap？
#   5. Transformer 的自注意力机制是怎么工作的？
# ============================================================
#  Chroma metadata 不接受`None` ，入库前必须删键
#  prompt 拼装时 context 必须是字符串（`"\n\n".join` ），messages 必须是字典列表
