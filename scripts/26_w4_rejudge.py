"""W5 卡3 附 | W4 A/C 组用新裁判（qwen3.8-max）重判，零生成成本。

为什么单独写脚本而不是重跑 `21_agentic_stats.py --judge`：
    21 会把 `data/eval_w4_agentic_stats_full.json` 【原地覆盖】，DeepSeek 判过的锚点就没了 ——
    而这次重判的**唯一价值就是把两代判官并排对照**，旧口径必须原样留着。
    轨迹 `w4_agentic_traces_full.jsonl` 里存了完整 answer 与 fused_hits（含 text），
    所以重判不需要重检索、更不需要重生成，一次生成 API 都不花。

口径与 21_agentic_stats.judge_anchor 逐字一致（直接复用它的函数，不复制实现）：
    · 只判 A / C 两组（B 是加固组、D≈C，判了只是第二次口径）；
    · 每题两跳：judge_correct（是否被资料支撑）+ judge_citations（逐条引用核查）；
    · cite_at_k 用 pooled，分母钉在送进 prompt的条数；out 且零引用的题不进分母。

落盘：data/eval_w4_agentic_rejudge_qwen.json（新旧两代 judge 的汇总 + 逐题翻转 + 指纹）

用法（项目根 rag-kb-next 下，需 .env 的 DASHSCOPE_API_KEY）：
    & "<venv>\\Scripts\\python.exe" scripts/26_w4_rejudge.py
    （结果已存在时需 --force）
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import (  # noqa: E402
    JUDGE_BASE_URL, JUDGE_MODEL, W4_AGENTIC_TRACE_JSONL,
)
from rag.gold import sha256_file  # noqa: E402

OUT_JSON = W4_AGENTIC_TRACE_JSONL.with_name("eval_w4_agentic_rejudge_qwen.json")
TRACE_FULL = W4_AGENTIC_TRACE_JSONL.with_name("w4_agentic_traces_full.jsonl")
# 21_agentic_stats.py 的落盘命名（STATS_TMPL）——它的旧锚点就是 DeepSeek 那一版，只读不写
OLD_STATS = W4_AGENTIC_TRACE_JSONL.with_name("eval_w4_agentic_stats_full.json")


def load_stats_module():
    """把 scripts/21_agentic_stats.py 当模块加载，复用它的判分聚合实现。

    scripts/ 不是包，importlib 是唯一不改目录结构的办法；
    复制一份 judge_anchor 到本文件会让「口径一致性」变成口头承诺 —— 一旦 21 改了口径，
    这里不会跟着变，两代数字就不可比。
    """
    spec = importlib.util.spec_from_file_location(
        "w4_stats", ROOT / "scripts" / "21_agentic_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if OUT_JSON.exists() and "--force" not in sys.argv:
        raise SystemExit(f"结果已存在：{OUT_JSON}\n要覆盖请加 --force（会再花约 80 次判分）")

    m = load_stats_module()
    rows = m.load_traces(TRACE_FULL)
    by_stage = m.index_by_stage(rows)
    records, meta = m.load_meta()
    missing = [st for st in m.STAGES if len(by_stage.get(st, {})) != len(records)]
    if missing:
        raise SystemExit(f"轨迹不完整：{missing} 的题数 ≠ {len(records)}，先补跑 20_agentic_ab.py")

    old_doc = json.load(open(OLD_STATS, encoding="utf-8"))
    old = old_doc.get("judge_anchor")
    if not old:
        raise SystemExit(f"{OLD_STATS.name} 里没有 judge_anchor，"
                         f"旧口径无从对照；先跑一次 21 --judge（用 DeepSeek）再重判")

    print(f"轨迹 {TRACE_FULL.name}（{len(rows)} 条，{len(records)} 题）")
    print(f"旧裁判：DeepSeek（来自 {OLD_STATS.name}）")
    print(f"新裁判：{JUDGE_MODEL}（thinking=off, temperature=0）；A/C 各 {len(records)} 题 × 2 跳\n")

    t0 = time.perf_counter()
    new = m.judge_anchor(by_stage, meta, stages=("A", "C"))
    elapsed = time.perf_counter() - t0

    # ---------------- 逐题翻转 ----------------
    flips = []
    for st in ("A", "C"):
        old_pq = {r["qid"]: r for r in (old["per_question"].get(st) or [])}
        for r in new["per_question"].get(st) or []:
            o = old_pq.get(r["qid"])
            if o is None:
                continue
            if bool(o["correct"]) != bool(r["correct"]):
                flips.append({"stage": st, "qid": r["qid"], "qtype": r.get("qtype"),
                              "old_correct": bool(o["correct"]), "new_correct": bool(r["correct"]),
                              "old_reason": o.get("correct_reason"),
                              "new_reason": r.get("correct_reason")})

    # ---------------- 汇总对照 ----------------
    def fmt(g: dict) -> dict:
        return {"n": g["n"], "correct": g["correct"], "answer_acc": g["answer_acc"],
                "cite_at_k_judge": g["cite_at_k_judge"],
                "cite_at_k_regex": g["cite_at_k_regex"],
                "zero_cite": g["zero_cite"]}

    groups = {}
    for st in ("A", "C"):
        groups[st] = {"old": fmt(old["groups"][st]), "new": fmt(new["groups"][st])}

    payload = {
        "experiment": "w4_rejudge_qwen",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "judge": {"old_model": old_doc.get("judge_anchor", {}).get("model") or "deepseek（原始锚点未记录模型名）",
                  "new_model": JUDGE_MODEL, "new_base_url": JUDGE_BASE_URL,
                  "temperature": 0, "extra_body": {"enable_thinking": False}},
        "fingerprints": {"trace_file": TRACE_FULL.name, "trace_sha256": sha256_file(TRACE_FULL),
                         "old_stats_file": OLD_STATS.name},
        "groups": groups,
        "flips": flips,
        "new_anchor": new,
        "old_anchor": old,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------------- 打印 ----------------
    print(f"{'组':<4}{'判官':<17}{'正确':<8}{'答案正确率':<12}{'cite_at_k':<12}{'零引用':<6}")
    for st in ("A", "C"):
        for tag in ("old", "new"):
            g = groups[st][tag]
            judge_name = "DeepSeek" if tag == "old" else JUDGE_MODEL
            print(f"{st:<4}{judge_name:<17}{g['correct']}/{g['n']:<6}"
                  f"{g['answer_acc'] * 100:<12.1f}{g['cite_at_k_judge'] * 100:<12.1f}{g['zero_cite']:<6}")
    print(f"\n翻转 {len(flips)} 条（旧→新）：")
    for x in flips:
        print(f"  {x['stage']} qid={x['qid']:<2} {x['qtype']:<9} "
              f"{x['old_correct']} → {x['new_correct']} | 旧：{x['old_reason']} | 新：{x['new_reason']}")
    print(f"\n判分 token（新）：{new['judge_tokens']}；耗时 {elapsed:.0f}s")
    print(f"已落盘：{OUT_JSON}")


if __name__ == "__main__":
    main()
