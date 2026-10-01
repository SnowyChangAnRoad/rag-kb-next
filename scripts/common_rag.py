"""
公共 RAG 模块 | Day8 抽出，08 及以后脚本 import 复用（Day9 再升级为 rag/ 包）。

职责边界：
- 本模块只做「组装」：连库 / 连 LLM / 检索 / 拼 prompt / 打印工具；
- 全项目唯一的嵌入函数仍定义在 common_embedding.py，这里只负责 import，绝不重写一份。

为什么不改 07：07 是已验收的历史脚本，保留原样作为「重构前」的对照；
两份 helper 的重复在 Day9 抽 rag/ 包时统一清理。

组合用法（08_rerank.py 就是这么用的）：
    from common_rag import get_client, get_collection, get_llm, retrieve, build_prompt, ...
"""
import os
import re
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
BATCH_SIZE = 64          # 入库分批大小

load_dotenv(str(ROOT / ".env"))   # 显式指定路径，不依赖当前工作目录

DEFAULT_K = 5            # 默认召回条数（08 精排时会显式传 10）

# ------------------------------------------------------------
# Prompt 模板（Day7 版本：强制真实编号 [k] 引用 + 有条件拒答）
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


def safe_preview(text: str, n: int = 40) -> str:
    """控制台只打印短预览，避免特殊 Unicode 触发 cp936 编码错误（Day3 教训）。"""
    return text.replace("\n", " ")[:n]


# ------------------------------------------------------------
# 连接类
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 数据 / 检索类
# ------------------------------------------------------------
def load_chunks() -> list[dict]:
    docs = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def retrieve(collection, question: str, k: int = DEFAULT_K) -> list[dict]:
    """向量召回：返回长度为 k 的 hits 列表，每项含 chunk_id/distance/text/元数据。"""
    output = collection.query(query_texts=[question], n_results=k)
    result = []
    for dist, doc, meta, cid in zip(
        output["distances"][0], output["documents"][0],
        output["metadatas"][0], output["ids"][0],
    ):
        result.append({"chunk_id": cid, "distance": dist, "text": doc, **meta})
    return result


def hit_where(hit: dict) -> str:
    """把一条 hit 的定位信息统一成「第x页」或 url，供各处打印/拼 prompt 共用。"""
    if hit.get("page") is not None:
        return f"第{hit['page']}页"
    return hit.get("url") or "网络资料"


def build_prompt(question: str, hits: list[dict]) -> str:
    """每块标【资料n】；回答里的 [n] 引用编号就指向这里的 n。"""
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(f"【资料{i}】来源：{hit['source']} {hit_where(hit)}\n{hit['text']}")
    context = "\n\n".join(blocks)
    return USER_TEMPLATE.format(context=context, question=question)


# ------------------------------------------------------------
# 打印类
# ------------------------------------------------------------
def format_sources(hits: list[dict]) -> str:
    """生成来源列表：每行形如 [1] RAG知识.pdf 第15页"""
    lines = [f"[{i}] {h['source']} {hit_where(h)}" for i, h in enumerate(hits, 1)]
    return "\n".join(lines) + "\n" if lines else ""


def show_hits(hits: list[dict]) -> None:
    """打印检索结果（交互模式用）。带 rank：该 hit 在召回列表中的原始名次。"""
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank} d={h['distance']:.4f}  {h['source']} {hit_where(h)}")
        print(f"      {safe_preview(h['text'])}")


# ------------------------------------------------------------
# 评测类（Day7 引入，08 的回归模式复用，保证与 Day7 结果可比）
# ------------------------------------------------------------
def extract_citations(answer: str) -> list[int]:
    """摘出回答里的引用编号：[2][1][3] → [1, 2, 3]（去重排序）。"""
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return sorted(set(nums))


def judge_answer(q: dict, answer: str, hits: list[dict]) -> tuple[bool, str]:
    """半自动判定：out 看拒答词；in/cross 看合法引用数；cross 额外要求来源≥2。"""
    qtype = q.get("type", "in")
    cites = [n for n in extract_citations(answer) if 1 <= n <= len(hits)]

    if qtype == "out":
        refuse_words = ["没有", "未在", "不涉及", "无法回答", "相关信息"]
        refused = any(w in answer for w in refuse_words)
        return refused, f"拒答词命中={refused}"

    ok_cite = len(cites) > 0
    note = f"cites={sorted(set(cites))}"
    if qtype == "cross":
        sources = {hits[n - 1]["source"] for n in cites}
        multi = len(sources) >= 2
        note += f", 引用来源数={len(sources)}"
        return ok_cite and multi, note
    return ok_cite, note