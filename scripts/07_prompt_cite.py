"""
Day 7 | 07_prompt_cite.py
目标：在 Day6 最小闭环基础上升级三件事——①system prompt 强制 [n] 引用 + 严格拒答；
      ②回答后打印「来源列表」，把 [n] 映射回 来源/页码/链接，便于核对引用是否正确；
      ③建 10 题测试集（5 库内 + 3 跨文档 + 2 库外），跑完用「半自动判定」分题打分。

用法（在项目根 rag-kb 下）：
    python scripts/07_prompt_cite.py build   # 全量入库（与 06 相同，幂等可重跑）
    python scripts/07_prompt_cite.py test    # 批量跑 data/test_questions.jsonl 的 10 题并判定
    python scripts/07_prompt_cite.py         # 交互问答（同上图，但多打「来源列表」），q 退出

链路（通关要能口述）：
    问题 → BGE 向量化 → Chroma top-k → 拼【资料n】上下文 → DeepSeek 回答(带[n]) → 来源列表

Day7 相比 Day6 的教学重点：
1. system prompt 是"可控的知识边界"：约束只能依据资料、强制引用编号、资料不足必须拒答。
   - 引用格式 [n] 中的 n 对应资料里的【资料n】（n 从 1 开始，n ≤ k=5），这就是"可追溯证据"。
   - 拒答 = 明确说"资料中没有相关内容"，而不是硬编一段。
2. 半自动判定：机器只做「客观/易错」的部分（有没有 [n]、有没有拒答词、是否跨文档多次引用），
   判断"引用到底对不对""答得对不对"仍要你人工复核——机器审格式，人审理据。
3. token 预算：k=5 的检索上下文 Day6 实测约 850–1030 token，今天只多加来源列表，量级不变；
   不要因为"多给了资料"就无限加 k，检索到的无关块会白白吃掉生成预算。

设计决策：
1. 沿用 Day6 的 day6_kb（不重建新库），build 仍是 upsert 幂等，是为了让 07 独立可跑。
2. 复用 06 的 helper（load_chunks/cmd_build/retrieve/…），不重复造轮子；Day9 才抽成 rag/ 包。
3. 判定规则尽量"宽容"：
   - out（库外）题：回答含"没有/未在资料中/不涉及/无法回答"等任一 → 通过（拒答成功）。
   - in（库内）/ cross（跨文档）：要求回答里出现 [n]；cross 再要求 ≥2 个不同编号（多跳整合）。
   - 引用编号不能超过检索到的块数（n ≤ len(hits)），否则是模型瞎编编号，判不通过。

硬边界：k=5；非流式（流式 Day12）；不做 rerank（Day8）。
"""
import os
import re
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
TEST_JSONL = ROOT / "data" / "test_questions.jsonl"

load_dotenv(str(ROOT / ".env"))   # 显式指定路径，不依赖当前工作目录

K = 5                # 检索条数（计划要求 k=5）
BATCH_SIZE = 64      # 入库分批大小

# ------------------------------------------------------------
# Prompt 模板（Day7 升级版：强制真实编号 [k] 引用 + 有条件拒答）
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一个基于个人知识库的问答助手，只能依据用户提供的资料回答问题。\n"
    "回答时，凡是结论、步骤、解释等来自某个【资料k】的内容，都要在该句后标注[k]；"
    "k 必须是对应资料的真实编号（1~5），例如由【资料2】支撑的句子就标 [2]，"
    "绝不要写成 [n]、[k] 这类占位符。如需用到多个资料，就分别标注各自编号。\n"
    "如果资料与问题完全无关、确实无法支撑任何回答，才明确说\"资料中没有相关内容\"，绝不编造；"
    "只要资料包含相关的概念、原理或步骤，就应基于资料作答并标注引用。\n"
    "用中文，简洁准确。"
)
USER_TEMPLATE = "资料：\n{context}\n\n问题：{question}"


# ------------------------------------------------------------
# 辅助函数（沿用 Day6，已给全）
# ------------------------------------------------------------
def safe_preview(text: str, n: int = 40) -> str:
    """控制台只打印短预览，避免特殊 Unicode 触发 cp936 编码错误（Day3 教训）。"""
    return text.replace("\n", " ")[:n]


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def get_llm() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def show_hits(hits: list[dict]) -> None:
    for rank, h in enumerate(hits, 1):
        if h.get("page") is not None:
            where = f"第{h['page']}页"
        else:
            where = h.get("url") or "?"
        print(f"  #{rank} d={h['distance']:.4f}  {h['source']} {where}")
        print(f"      {safe_preview(h['text'])}")


def load_chunks() -> list[dict]:
    docs = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def cmd_build() -> None:
    chunks = load_chunks()
    print(f"读入 {len(chunks)} 块，目标库：{COLLECTION_NAME}（cosine）")

    def clean_meta(meta: dict) -> dict:
        return {k: v for k, v in meta.items() if v is not None}

    collection = get_collection(get_client())
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        metadatas = [
            clean_meta({             # 去掉值为None的键，因为collection.upsert不接受空值
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "source": c["source"],
                "page": c["page"],
                "url": c["url"],
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


# 召回TOP-K文段，返回hits
def retrieve(collection, question: str, k: int = K) -> list[dict]:
    # query 默认返回 ids、documents、metadatas 和 distances 为 键的字典
    output = collection.query(query_texts=[question], n_results=k)
    result = []
    # zip 的作用是把多个可迭代对象按位置一一配对，返回一个元组的迭代器。
    for (dist, doc, meta, id) in zip(
        output["distances"][0], output["documents"][0], output["metadatas"][0], output["ids"][0]
    ):
        result.append({"chunk_id": id, "distance": dist, "text": doc, **meta})
    return result


def build_prompt(question: str, hits: list[dict]) -> str:
    """每块标【资料n】；[n] 引用编号就指向这里的 n。沿用 Day6，无需改。"""
    blocks = []
    for i, hit in enumerate(hits, 1):
        if hit.get("page") is not None:
            where = f"第{hit['page']}页"
        else:
            where = hit.get("url") or "网络资料"
        blocks.append(f"【资料{i}】来源：{hit['source']} {where}\n{hit['text']}")
    context = "\n\n".join(blocks)
    return USER_TEMPLATE.format(context=context, question=question)


def ask_once(collection, llm: OpenAI, question: str, verbose: bool = True) -> tuple[str, str, list[dict]]:
    """单次问答。返回 (回答文本, 来源列表字符串, hits)。交互模式 verbose=True，
    test 模式 verbose=False（逐题自己排版）。"""
    hits = retrieve(collection, question, K)
    if verbose:
        print("--- 检索结果 ---")
        show_hits(hits)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    resp = llm.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.7,   # 保持 Day6 默认；测试如果想要更可复现，可临时调低
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = resp.choices[0].message.content
    if verbose:
        print("--- 回答 ---")
        print(answer)
        print("--- 来源列表 ---")
        print(format_sources(hits))
        print(f"Token 用量：输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens}")
    return answer, format_sources(hits), hits


# ============================================================
# TODO 1：摘出回答里的引用编号，约 3 行
#   输入：answer（模型回答文本）
#   输出：去重排序的整数列表，如 "[2][1][3] 和 [1]" → [1, 2, 3]
#   提示：re.findall(r"\[(\d+)\]", answer) 拿到的是字符串，要 int()；
#         若没有任何引用，返回空列表 []
# ============================================================
def extract_citations(answer: str) -> list[int]:
    # 关键点：findall 取回的是字符串，必须先 int() 再排序，
    # 否则 judge_answer 里 max(cites) <= len(hits) 会 str 与 int 比较报 TypeError
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return sorted(set(nums))   # 去重+排序


# ============================================================
# TODO 2：生成「来源列表」字符串，约 8 行
#   输入：hits（retrieve 返回的 k 条）
#   输出：多行字符串，每行形如 [1] RAG知识.pdf 第15页 / [2] RAG 技术概览.html https://...
#   提示：顺序与【资料n】一致（hits[0] 就是【资料1】）；page 有就"第x页"，
#         否则用 url（兜底"网络资料"）；每行前加序号
# ============================================================
def format_sources(hits: list[dict]) -> str:
    # TODO 2 你填
    output = ""
    for i, hit in enumerate(hits,1):
        if hit.get("page") is not None:
            where = f"第{hit["page"]}页"
        else:
            where = hit.get("url") or "网络资料"
        output += f"[{i}] {hit.get("source")} {where}\n"
    return output


# ============================================================
# TODO 3：半自动判定
#   输入：q（dict，含 type）、answer（回答）、hits（检索到的 k 条）
#   输出：(ok: bool, note: str)
#   规则（宽容）：
#     out 题：回答含任一拒答词 → 通过；否则（开始详细介绍内容）→ 不通过
#            拒答词参考：["没有", "未在", "不涉及", "无法回答", "相关信息"]
#     in 题 ：extract_citations 有引用，且最大编号 ≤ len(hits) → 通过
#     cross ：在 in 基础上，还要 去重后的引用数 ≥ 2（多跳整合）→ 通过
#   note 写简短原因，如 {"cites":[1,2], "max_ok":True}，方便你人工复核
# ============================================================
def judge_answer(q: dict, answer: str, hits: list[dict]) -> tuple[bool, str]:
    qtype = q.get("type", "in")
    # 引用编号先过滤掉越界的（如 [9] 超过 len(hits)=5 就是模型瞎编的编号）
    cites = [n for n in extract_citations(answer) if 1 <= n <= len(hits)]

    if qtype == "out":        # 库外题：看有没有明确拒答
        refuse_words = ["没有", "未在", "不涉及", "无法回答", "相关信息"]
        refused = any(w in answer for w in refuse_words)
        return refused, f"拒答词命中={refused}"

    ok_cite = len(cites) > 0   # 必须出现了合法引用
    note = f"cites={sorted(set(cites))}"
    if qtype == "cross":       # 多跳整合：被引用的块须来自 ≥2 个不同来源，才算真·跨文档
        sources = {hits[n - 1]["source"] for n in cites}
        multi = len(sources) >= 2
        note += f", 引用来源数={len(sources)}"
        return ok_cite and multi, note
    return ok_cite, note


# ============================================================
# TODO 4：批量跑测试集
#   1) 读 TEST_JSONL（逐行 json.loads，跳过空行）
#   2) 对每题：ask_once(verbose=False) 拿回答 → judge_answer 判 ok/note
#   3) 逐题打印一页：id/type/问题/来源列表/回答(前若干字)/判定 ✅ 或 ❌ + note
#   4) 最后汇总打印：通过数/总数，并按 type 分开统计（如 "in 4/5, cross 2/3, out 2/2"）
#   小技巧：问题长就先打前 30 字 + "…"，回答太长也只打前 80 字，方便一屏扫完
# ============================================================
def cmd_test() -> None:
    questions = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"读入 {len(questions)} 道测试题，开始批量问答…")

    collection = get_collection(get_client())
    llm = get_llm()

    # 按 type 统计 [通过数, 总数]，主循环逐题打印
    stats: dict[str, list[int]] = {"in": [0, 0], "cross": [0, 0], "out": [0, 0]}
    for q in questions:
        qid, qtype, question = q["id"], q["type"], q["question"]
        answer, sources, hits = ask_once(collection, llm, question, verbose=False)
        ok, note = judge_answer(q, answer, hits)
        stats[qtype][0] += int(ok)
        stats[qtype][1] += 1
        mark = "✅" if ok else "❌"
        print("\n" + "=" * 60)
        print(f"[{qid}] ({qtype}) 判定 {mark}  {note}")
        print(f"问题：{safe_preview(question, 50)}")
        print("--- 来源列表 ---")
        print(sources.rstrip() or "(无来源)")
        print(f"--- 回答（前 120 字）---")
        print(safe_preview(answer, 120))

    total_ok = sum(v[0] for v in stats.values())
    total = sum(v[1] for v in stats.values())
    print("\n" + "=" * 60)
    print(f"汇总：通过 {total_ok}/{total}")
    for t in ("in", "cross", "out"):
        print(f"  {t}: {stats[t][0]}/{stats[t][1]}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "build":
        cmd_build()
        return
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        cmd_test()
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
            ask_once(collection, llm, q, verbose=True)
        return
    print("用法：python scripts/07_prompt_cite.py [build|test]")
    sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================
# ------------------------架构图-------------------------------
# 1.build:
#   cmd_build()读入json数据，存入collection数据库
# 2.test:
#   cmd_test()
#     读入 questions.json 测试题
#     for q in questions:
#         ask_once(collection, llm, question, verbose=False)
#             retrieve(collection, question, K)  获取hits
#                 collection.query()  召回TOP-K
#             build_prompt()+使用模板prompt  构建message
#             llm.chat.completions.create()  向大模型发送请求，获取回复
#         judge_answer(q, answer, hits)  自动化评判结果
#         输出结果和评判结果
# 3.main() _:
#   main() 交互输入问题，获取大模型回复
# ============================================================
# 知识五问（原理层，非结果观察；答案已批改订正，9/17 夜）
# ------------------------------------------------------------
# K1. 为什么模型会输出字面 [n1]？改了什么才修好？
#     机制：LLM 没有「元语言层」，prompt 里所有字符对它都只是上下文 token 序列，
#           它的本能是「续写看到的模式」。写「格式为[n]」时，[n] 这个字面模式就落在
#           上下文里，模型直接复制它，而不是把 n 替换成真实编号。
#     注意因果：修好靠的不是「解释这是描述还是指令」（模型同样分不清这种元语言），
#           而是「消除空符号 + 给真实样本」——把 [n] 换成实值 [2]，并明令禁写占位符，
#           让它无抽象符号可复制，只有具体编号可模仿。
# K2. 加引用为什么能减少幻觉？全量引用 [1,2,3,4,5] 算成功吗？
#     机制：加引用把任务从「生成」变成「生成 + 归因」，每个断言都必须回指到输入证据，
#           等于在解码时加了指向上下文的约束，压缩了「无依据生成」这条路径（grounding）。
#     全量引用不是成功而是退化：标了 5 个编号就无法从标注看出「哪句出自哪块」，
#           引用失去区分力，沦为装饰性后缀。
#     纠偏：「引用偏少=更精确」不成立——少可能是精确，也可能是漏引/答不全。
#           正确标准是「引用覆盖与实际用到的证据量相匹配」。
# K3. judge_answer 的 out 判定（查拒答词表）有什么漏洞？
#     本质：词表匹配的是「表面字符串」，与语境无关；而拒答是「语义行为」。
#     误判例：「若没有 RAG 的辅助，模型幻觉率较高」——含「没有」→ 词表命中 → 判拒答✅，
#           可它根本没拒答，是在正常答题。同一个「没有」在「资料中没有相关内容」里是
#           拒答标志，在「若没有 X」里只是假设连词，判定器看不见这个差别。
#     反向漏判：模型换说法拒答（如「无法从上述材料得出」）而词表未覆盖 → 被判❌。
# K4. cross 判定到底测什么？（当前实现：被引用块来自 ≥2 个不同 source）
#     它测的不是「有没有综合」，只是「引用来源的分散度」。
#     反例：「向量检索按余弦距离排序[1]。另外 RAG 能降低幻觉[2]。」——引了两个来源，
#           两句话各说各的、毫无整合，照样判✅。
#     真正的多跳应测「证据必要性」：删掉任一来源的块，答案是否就不再成立。
#           实操可用 LLM-as-judge 做「必要性检验」→ Day9 评测升级方向。
# K5. 为什么问「overlap」召回不到 p76（原文是「考虑调整块的重叠大小…」）？
#     机制：bi-encoder 把 query 和 doc 各自独立编码，检索时只做向量点积。
#           编码 query 时看不到 doc、编码 doc 时也看不到 query，没有机会做词汇对齐
#           （把英文 overlap 与中文「重叠」对上），同义不同形的表达在向量空间距离远。
#     对照：cross-encoder 把 query+doc 拼成一个序列送进同一 Transformer，token 间做
#           full attention，能做细粒度交互（连否定、词序都能处理），质量更高。
#     为什么只能用于重排：cross-encoder 无法预缓存（N 篇文档 × 无限 query 组合爆炸），
#           所以只对少量候选（top10）做精排。
#     → 这就是「召回求快求全 / 精排求准」两阶段架构的根本原因，也是 Day8 rerank 的立论根据。