"""入口层 | Day9：一行命令完成问答

用法（在项目根 rag-kb 下执行）：
    python -m rag.cli                     # 交互模式：输入问题回车，q 退出；答完会问 👍/👎
    python -m rag.cli "为什么加 overlap"    # 单次问答（脚本化调用，不问反馈）

对照：以前要 `python scripts/06_naive_rag.py`（无精排）或 `python scripts/08_rerank.py`
（精排但无去重、无结构化返回）；现在是同一段代码一次调用，且返回里带耗时/token/成本。

Day10 卡 5：交互模式每次问答都会写一条 trace（data/traces/traces.jsonl），
答完按 1/2 可直接落一条 👍/👎 反馈；补录与查看走 python -m rag.trace。
"""
import sys

from .config import COLLECTION_NAME, RERANK_TOP_K
from .llm import safe_preview
from .pipeline import RAGPipeline
from .retriever import get_client, get_collection
from .trace import Tracer, add_score


def print_result(out: dict, question: str) -> None:
    """已给：把 pipeline 的结构化结果打印成人看的样子。"""
    print(f"\n问题：{question}")
    print(f"--- 进入 prompt 的资料：{out['top_k']} 条"
          f"（候选池 {out['n_recall']} → 去重后 {out['n_dedup']}）---")
    for i, h in enumerate(out["hits"], 1):
        score = h.get("rerank_score")
        score_txt = f"s={score:.4f} " if score is not None else ""
        print(f"  【资料{i}】{score_txt}d={h['distance']:.4f} {h['source']} "
              f"{h.get('page') and '第' + str(h['page']) + '页' or h.get('url') or ''}")
        print(f"      {safe_preview(h['text'], 44)}")
    print("--- 回答 ---")
    print(out["answer"])
    print("--- 来源列表 ---")
    print(out["sources"].rstrip() or "(无来源)")
    print(f"耗时：检索 {out['retrieve_ms']:.0f}ms（召回+精排+去重） 生成 {out['gen_ms']:.0f}ms")
    print(f"token：输入 {out['prompt_tokens']} / 输出 {out['completion_tokens']}"
          f"  成本 ≈ ¥{out['cost']:.4f}")
    if out.get("trace_id"):
        print(f"trace：{out['trace_id']}   ← python -m rag.trace show {out['trace_id']}")


def ask_feedback(out: dict) -> None:
    """已给：交互模式里采一条 👍/👎（Day10 卡 5 的 score；回车跳过）。

    只在交互模式调用 —— 单次问答常被脚本/评测复用，插一个 input() 会卡住调用方。
    """
    trace_id = out.get("trace_id")
    if not trace_id:
        return
    try:
        answer = input("这条回答有用吗？[1]👍 [2]👎 [回车]跳过 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    value = {"1": "up", "2": "down"}.get(answer)
    if value:
        add_score(trace_id, value)
        print(f"已记录反馈 {value} → {trace_id}")


def main() -> None:
    collection = get_collection(get_client())
    tracer = Tracer()                       # Day10 卡 5：cli 的每次问答都埋点
    pipeline = RAGPipeline(collection)
    print(f"已加载 {collection.count()} 条知识块（{COLLECTION_NAME}）；"
          f"链路：向量召回 → bge-reranker 精排 → 相邻块去重 → 截断 top{RERANK_TOP_K} → 生成")

    # 命令行直接带问题 = 单次问答（脚本化调用，也是 Day11 包 API 的雏形）
    one_shot = " ".join(sys.argv[1:]).strip()
    if one_shot:
        print_result(pipeline.answer(one_shot, tracer=tracer), one_shot)
        return

    while True:
        try:
            question = input("\n你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            break
        out = pipeline.answer(question, tracer=tracer)
        print_result(out, question)
        ask_feedback(out)


if __name__ == "__main__":
    main()