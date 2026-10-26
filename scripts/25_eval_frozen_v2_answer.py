"""W5 卡3 | frozen_v2 生成层评测（DeepSeek 作答 + 百炼 qwen 判分）。

要回答的问题：30 道冻结题在【交付口径】上「答得对不对、引用用得好不好」，
以及这套生成层数字能否摆脱「judge 与被测同源」的原罪。

与 23_eval_frozen_v2.py 的分工（不要混）：
    23 = 检索层，零 API、可复现，pytest 门禁的对照物（recall / MRR / nDCG）；
    本脚本 = 生成层，花钱、有抖动，只出报告、不进默认 CI。

口径（每一项都能在报告里对上）：
    作答：fact 25 题走标准 RAG（dense + rerank + dedupe + quota_cap=2 + top_k=3 + prompt v2），
          与 api/main.py 的 /chat 默认值逐项一致；
          principle 5 题不检索（hits=[]），走 v1-noretr 出口（与 W4 gate 判「不检索」同一条路径）。
    判分：fact 题 = 裁判 1（是否被资料支撑）+ 裁判 2（逐条引用核查）；
          principle 题 = 裁判 3（reference 要点覆盖率，≥60% 记正确）。
    指标：答案正确率按【全部 30 题】算；cite_at_k 只按 fact 且写过引用的题算（pooled 口径，
          零引用题不出现在分母里）；principle 的覆盖率单列，不混进 cite_at_k。

用法（项目根 rag-kb-next 下，需 .env 里同时有 DEEPSEEK_API_KEY 与 DASHSCOPE_API_KEY）：
    & "<venv>\\Scripts\\python.exe" scripts/25_eval_frozen_v2_answer.py            # 全跑
    & "<venv>\\Scripts\\python.exe" scripts/25_eval_frozen_v2_answer.py --limit 5  # 冒烟
    （结果文件已存在时默认拒绝覆盖 —— 重跑是一笔钱，加 --force 是显式决策）

落盘：data/frozen_v2_generation.json（指纹 + 判官身份 + 逐题全文答案与判分明细 + 汇总）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import (  # noqa: E402
    CHUNKS_JSONL, FROZEN_V2_GENERATION_JSON, FROZEN_V2_JSONL, GOLD_FROZEN_V2_JSON,
    JUDGE_BASE_URL, JUDGE_EXTRA_BODY, JUDGE_MODEL, JUDGE_PRINCIPLE_PASS, LLM_MODEL,
    RETRIEVAL_MODE, RERANK_TOP_K, SOURCE_CAP,
)
from rag.eval import count_valid_cited, extract_cited_ids  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.judge import judge_citations, judge_correct, judge_principle  # noqa: E402
from rag.llm import ask, get_llm, safe_preview  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.config import resolve_prompt  # noqa: E402
from rag.retriever import get_client, get_collection  # noqa: E402

FACT_PROMPT = "v2"           # 交付口径（与 /chat 默认一致）
PRINCIPLE_PROMPT = "v1-noretr"   # 不检索出口（与 W4 gate_no_retrieve 同一条路径）


# ------------------------------------------------------------
# 装载：题目 + gold 指纹（与 23_eval_frozen_v2.load_records 同口径）
# ------------------------------------------------------------
def load_records() -> list[dict]:
    """读 frozen_v2.jsonl 与 gold，并校验两边指纹。

    为什么两个脚本各有一份装载逻辑：scripts/ 不是包，跨脚本 import 需要改 sys.path 动手术；
    这 20 行的重复换的是「两个入口都不依赖对方的存在」。指纹校验必须两处都做 ——
    少做一处，旧 gold 就会在那条路径上静默生效。
    """
    with open(FROZEN_V2_JSONL, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    payload = load_gold(GOLD_FROZEN_V2_JSON)      # 内部已校验题目 sha256
    chunks_sha = sha256_file(CHUNKS_JSONL)
    if payload["chunks_sha256"] != chunks_sha:
        raise RuntimeError(
            f"语料已变：gold 锚定 {payload['chunks_sha256'][:12]}… "
            f"≠ 当前 {chunks_sha[:12]}… —— 重跑 python -m rag.gold frozen_v2"
        )

    gold_by_id = {r["id"]: r for r in payload["records"]}
    q_ids = {q["id"] for q in questions}
    if q_ids != set(gold_by_id):
        only_q = sorted(q_ids - set(gold_by_id))
        only_g = sorted(set(gold_by_id) - q_ids)
        raise RuntimeError(f"题目与 gold 的 id 对不上：仅题 {only_q} / 仅 gold {only_g}")

    return [{**q, **{k: v for k, v in gold_by_id[q["id"]].items()
                     if k in ("unit", "gold", "detail")}}
            for q in questions]


# ------------------------------------------------------------
# 作答
# ------------------------------------------------------------
def answer_fact(pipeline: RAGPipeline, q: dict) -> dict:
    """标准 RAG 路径（参数取服务默认值，逐项写死便于报告复述）。"""
    out = pipeline.answer(q["question"], qtype=q["type"], dedupe=True,
                          top_k=RERANK_TOP_K, quota_cap=SOURCE_CAP,
                          prompt_version=FACT_PROMPT, mode=RETRIEVAL_MODE)
    return {"answer": out["answer"], "hits": out["hits"], "top_k": out["top_k"],
            "prompt_version": out["prompt_version"], "retrieval_mode": out["retrieval_mode"],
            "retrieve_ms": out["retrieve_ms"], "gen_ms": out["gen_ms"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"], "cost": out["cost"]}


def answer_principle(llm, q: dict) -> dict:
    """不检索出口：hits=[] + v1-noretr。检索段耗时为 0（这题本来就不该走检索）。"""
    t0 = time.perf_counter()
    gen = ask(llm, q["question"], [], resolve_prompt(PRINCIPLE_PROMPT))
    gen_ms = (time.perf_counter() - t0) * 1000
    return {"answer": gen["answer"], "hits": [], "top_k": 0,
            "prompt_version": PRINCIPLE_PROMPT, "retrieval_mode": "none",
            "retrieve_ms": 0.0, "gen_ms": gen_ms,
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"], "cost": gen["cost"]}


# ------------------------------------------------------------
# 判分
# ------------------------------------------------------------
def judge_fact(q: dict, out: dict) -> dict:
    jc = judge_correct(q["question"], out["hits"], out["answer"])
    jcit = judge_citations(q["question"], out["hits"], out["answer"])
    cited = extract_cited_ids(out["answer"])
    valid_ids = count_valid_cited(cited, jcit["invalid"], out["top_k"])
    return {"correct": jc["correct"], "reason": jc["reason"],
            "cite_total": jcit["total"], "cite_valid": jcit["valid"],
            "cite_invalid": jcit["invalid"], "cite_missing": jcit["missing"],
            "cited_ids": cited, "valid_cited_ids": valid_ids,
            "judge_usage": [jc["usage"], jcit["usage"]],
            "judge_cost": jc["cost"] + jcit["cost"]}


def judge_principle_item(q: dict, out: dict) -> dict:
    jp = judge_principle(q["question"], q["reference"], out["answer"])
    return {"correct": jp["correct"], "reason": jp["reason"],
            "coverage": jp["coverage"], "covered": jp["covered"], "n_points": jp["total"],
            "judge_usage": [jp["usage"]], "judge_cost": jp["cost"]}


# ------------------------------------------------------------
# 汇总
# ------------------------------------------------------------
def summarize(records: list[dict]) -> dict:
    """分母按题型分开写清楚：correct 用全部题；cite 只用「应作答」的 fact（in/cross/partial）；
    principle 只看覆盖率。

    为什么 out 题不进引用分母：out 考的是「该不该答」，不是「引得对不对」。部分拒答的 out 题
    （如 26/28）会引用库内资料来解释「哪些部分资料没有」，把这些引用算进 cite_acc 会奖励
    「答了库外题」；且模型常用全角【资料n】，与半角 [n] 抽取口径不一致（29 号实测 3 vs 0）。
    Day14 的 out 题恰好全部零引用，所以两种口径等价；这里把规则写死，不留隐含假设。
    """
    fact = [r for r in records if r["kind"] == "fact"]
    answerable = [r for r in fact if r["type"] != "out"]   # 引用分母只算应作答的题
    principle = [r for r in records if r["kind"] == "principle"]

    cite_total = sum(r["cite_total"] for r in answerable)
    cite_valid = sum(r["cite_valid"] for r in answerable)
    slot_valid = sum(len(r["valid_cited_ids"]) for r in answerable)
    slot_total = sum(r["top_k"] for r in answerable)

    by_type: dict[str, dict] = {}
    for t in dict.fromkeys(r["type"] for r in records):
        rs = [r for r in records if r["type"] == t]
        by_type[t] = {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                      "rate": sum(r["correct"] for r in rs) / len(rs)}

    judge_prompt = sum(u["prompt_tokens"] for r in records for u in r["judge_usage"])
    judge_completion = sum(u["completion_tokens"] for r in records for u in r["judge_usage"])

    return {
        "n": len(records),
        "answer_acc": sum(r["correct"] for r in records) / len(records),
        "by_type": by_type,
        "fact_n": len(fact),
        "cite_n": len(answerable),
        "cite_acc": (cite_valid / cite_total) if cite_total else None,
        "cite_total": cite_total, "cite_valid": cite_valid,
        "cite_at_k": (slot_valid / slot_total) if slot_total else None,
        "slot_valid": slot_valid, "slot_total": slot_total,
        "zero_cite_n": sum(1 for r in fact if r["cite_total"] == 0),
        "principle_n": len(principle),
        "principle_pass": sum(r["correct"] for r in principle),
        "principle_coverage_mean": (sum(r["coverage"] for r in principle) / len(principle)
                                    if principle else None),
        "avg_retrieve_ms": (sum(r["retrieve_ms"] for r in fact) / len(fact) if fact else 0.0),
        "avg_gen_ms": sum(r["gen_ms"] for r in records) / len(records),
        "gen_prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "gen_completion_tokens": sum(r["completion_tokens"] for r in records),
        "gen_cost": sum(r["cost"] for r in records),
        "judge_prompt_tokens": judge_prompt,
        "judge_completion_tokens": judge_completion,
        "judge_cost": sum(r["judge_cost"] for r in records),
    }


def print_summary(s: dict) -> None:
    print(f"\n{'=' * 88}\nfrozen_v2 生成层（judge = {JUDGE_MODEL}，temperature=0）\n")
    print(f"  答案正确率 {s['answer_acc'] * 100:.1f}%  "
          f"({sum(v['correct'] for v in s['by_type'].values())}/{s['n']})")
    print("  分题型：" + "  ".join(f"{t} {v['correct']}/{v['n']}"
                                   for t, v in s["by_type"].items()))
    if s["cite_at_k"] is not None:
        print(f"  引用正确率 {s['cite_acc'] * 100:.1f}%  "
              f"({s['cite_valid']}/{s['cite_total']}，in/cross/partial、零引用不计分母）")
        print(f"  引用覆盖率 {s['slot_valid']}/{s['slot_total']} = "
              f"{s['cite_at_k'] * 100:.1f}%（pooled，分母 = top_k 槽位）")
        print(f"  零引用题 {s['zero_cite_n']} / {s['fact_n']}（fact）")
    if s["principle_n"]:
        print(f"  principle 覆盖率均值 {s['principle_coverage_mean'] * 100:.1f}%  "
              f"通过 {s['principle_pass']}/{s['principle_n']}（阈值 ≥60%）")
    print(f"  平均检索 {s['avg_retrieve_ms']:.0f}ms / 平均生成 {s['avg_gen_ms']:.0f}ms")
    print(f"  生成 token {s['gen_prompt_tokens']}/{s['gen_completion_tokens']}，"
          f"¥{s['gen_cost']:.4f}；判分 token {s['judge_prompt_tokens']}/"
          f"{s['judge_completion_tokens']}，¥{s['judge_cost']:.4f}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用，0=全跑）")
    ap.add_argument("--force", action="store_true", help="结果文件已存在时允许覆盖")
    args = ap.parse_args()

    # 冒烟落独立文件：否则 --limit 3 写出的「正式结果」会挡住全量跑（还得 --force）
    out_path = (FROZEN_V2_GENERATION_JSON if not args.limit
                else FROZEN_V2_GENERATION_JSON.with_name(
                    "frozen_v2_generation_smoke.json"))
    if out_path.exists() and not args.force and not args.limit:
        raise SystemExit(
            f"结果已存在：{out_path}\n"
            f"重跑会再花一次 API 钱 —— 确认要覆盖就加 --force"
        )

    records_in = load_records()
    if args.limit:
        records_in = records_in[:args.limit]

    collection = get_collection(get_client())
    pipeline = RAGPipeline(collection)
    deepseek = get_llm()

    print(f"库 {collection.name}（{collection.count()} 块）；{len(records_in)} 题；"
          f"作答 DeepSeek → 判分 {JUDGE_MODEL}")
    print(f"作答参数：dedupe=True, top_k={RERANK_TOP_K}, quota_cap={SOURCE_CAP}, "
          f"mode={RETRIEVAL_MODE}, prompt={FACT_PROMPT}/{PRINCIPLE_PROMPT}")

    records: list[dict] = []
    for q in records_in:
        if q["kind"] == "principle":
            out = answer_principle(deepseek, q)
            judged = judge_principle_item(q, out)
        else:
            out = answer_fact(pipeline, q)
            judged = judge_fact(q, out)
        rec = {"id": q["id"], "type": q["type"], "kind": q["kind"],
               "question": q["question"], **out, **judged}
        records.append(rec)
        mark = "✅" if rec["correct"] else "❌"
        if rec["kind"] == "fact":
            cite = (f"引用 {rec['cite_valid']}/{rec['cite_total']}"
                    if rec["cite_total"] else "无引用")
            cover = f"覆盖 {len(rec['valid_cited_ids'])}/{rec['top_k']}"
        else:
            cite, cover = "不检索", f"覆盖率 {rec['coverage'] * 100:.0f}%"
        print(f"  [{rec['id']:>2}] {mark} {rec['type']:<9} {cite:<10} {cover:<12} "
              f"| {safe_preview(rec['question'], 26)}")

    summary = summarize(records)
    result = {
        "experiment": "frozen_v2_generation_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge": {"model": JUDGE_MODEL, "temperature": 0,
                  "base_url": os.getenv("DASHSCOPE_BASE_URL") or JUDGE_BASE_URL,
                  "extra_body": JUDGE_EXTRA_BODY,
                  "principle_pass_threshold": JUDGE_PRINCIPLE_PASS},
        "generation": {"model": LLM_MODEL, "mode": RETRIEVAL_MODE,
                       "dedupe": True, "top_k": RERANK_TOP_K, "quota_cap": SOURCE_CAP,
                       "prompt": {"fact": FACT_PROMPT, "principle": PRINCIPLE_PROMPT}},
        "fingerprints": {"frozen_v2_sha256": sha256_file(FROZEN_V2_JSONL),
                         "chunks_sha256": sha256_file(CHUNKS_JSONL)},
        "summary": summary,
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print_summary(summary)
    print(f"\n明细已落盘：{out_path}")


if __name__ == "__main__":
    main()
