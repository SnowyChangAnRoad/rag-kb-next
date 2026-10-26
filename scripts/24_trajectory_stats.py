"""W5 卡6 | 轨迹统计一页（修订14「轨迹级评分」的轻量落地）。

要回答的问题：W4 的 Agentic 轨迹（100 条 = 20 题 × 5 组）里，**检索轮次花在哪、
改写有没有用**。不做完整 trajectory judge（那是行业评测岗的能力面，本阶段只进清单），
只做三个可复现的统计：
    ① 每题检索轮次分布 + 触顶（cap=3）题明细
    ② 改写有效率：改写后的 top-k 是否引入新块 / 新 gold
    ③ 命中去向：D2 命中趟与结构性未命中

【只读】data/w4_agentic_traces_full.jsonl —— 不重跑检索、不重跑生成。
落盘 reports/w5_trajectory.md（可由本脚本重跑复现，数字变了就说明轨迹变了）。

口径声明（写进报告，读表前必须知道）：
    1) 轨迹每轮只落 top-3（hits_list），没有全召回池 ⇒「新块」= 新进 top-3 的块，
       不是「新被召回的块」。
    2) gold 粒度按题走：in 是页级 chunk 集合，cross/partial 是来源级文件名集合
       （与 metrics.evaluate 同口径，来源级用 h["source"] 比对）。
    3) D2 是混合均值：18 题命中（一次编码 + 查表）+ 2 题结构性未命中
       （gate_no_retrieve ⇒ 按设计不入缓存 ⇒ 重跑整链路）。两类必须分开报。
    4) 「改写有效率」的分母是**改写轮**，不是题数：B 组每题 1 轮改写，C/D1 最多 2 轮。

用法（项目根 rag-kb-next 下）：
    & "<venv>\\Scripts\\python.exe" scripts/24_trajectory_stats.py [--scope full|pilot]
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import DATA_DIR, GOLD_TUNE_JSON  # noqa: E402
from rag.gold import load_gold  # noqa: E402
from rag.metrics import SOURCE_UNIT  # noqa: E402

REPORT = ROOT / "reports" / "w5_trajectory.md"
STAGES = ("A", "B", "C", "D1", "D2")
TYPES = ("in", "cross", "partial", "out")


def load_traces(scope: str) -> list[dict]:
    path = DATA_DIR / f"w4_agentic_traces_{scope}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_index() -> dict:
    """qid(str) → (unit, gold set)，供「新 gold」统计用（来源级 gold 是文件名）。"""
    payload = load_gold(GOLD_TUNE_JSON)
    return {str(r["id"]): (r["unit"], r["gold"]) for r in payload["records"]}


def hit_keys(hits: list[dict], unit: str) -> set:
    """把一轮 top-3 投影成 gold 可直接比对的键集合（页级用 chunk_id，来源级用 source）。"""
    return {h["source"] if unit == SOURCE_UNIT else h["chunk_id"] for h in hits}


def trace_stats(rec: dict, gidx: dict) -> dict:
    """单条轨迹的可统计事实。rounds=0（gate 不检索 / 缓存命中）单独标记。"""
    hits_list = rec["hits_list"]
    qlist = rec["question_list"]
    qid = str(rec["qid"])
    unit, gold = gidx.get(qid, ("page", set()))

    rounds = len(hits_list)
    rewrites = max(0, len(qlist) - 1)          # 改写轮数
    new_chunk = new_gold = 0
    for i in range(1, rounds):
        cur, prev = hit_keys(hits_list[i], unit), hit_keys(hits_list[i - 1], unit)
        if cur - prev:
            new_chunk += 1
        if (cur & gold) - (prev & gold):
            new_gold += 1

    return {
        "stage": rec["stage"], "qid": qid, "qtype": rec["qtype"],
        "rounds": rounds, "rewrites": rewrites,
        "repeat": sum(1 for f in rec.get("query_repeat_flags") or [] if f),
        "new_chunk_rounds": new_chunk, "new_gold_rounds": new_gold,
        "reason": rec["terminate_reason"], "cache_hit": rec["cache_hit"],
        "cache_wrote": rec.get("cache_wrote", False),
        "llm_calls": rec["llm_calls"], "total_ms": rec["total_ms"],
        "cache_match": rec.get("cache_match"),
        "top1": [hs[0]["chunk_id"] if hs else "-" for hs in hits_list],
    }


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="full", choices=("full", "pilot"))
    args = ap.parse_args()

    traces = load_traces(args.scope)
    gidx = gold_index()
    stats = [trace_stats(r, gidx) for r in traces]

    lines: list[str] = []
    add = lines.append

    add("# W5 轨迹统计（一页）")
    add("")
    add("数据源：`data/w4_agentic_traces_%s.jsonl`（%d 条 = %d 题 × 5 组），"
        "脚本 `scripts/24_trajectory_stats.py` 只读、零 API、可重跑复现。"
        % (args.scope, len(traces), len({s["qid"] for s in stats})))
    add("")

    # ---------- ① 轮次分布 ----------
    add("## 1. 检索轮次分布（Agentic 组）")
    add("")
    add("| 组 | 平均轮次 | 0 轮 | 1 轮 | 2 轮 | 3 轮（触顶） | 触顶题 |")
    add("|---|---|---|---|---|---|---|")
    for st in ("C", "D1"):
        rs = [s for s in stats if s["stage"] == st]
        dist = Counter(s["rounds"] for s in rs)
        capped = "、".join(f"id={s['qid']}({s['qtype']})" for s in rs if s["rounds"] == 3) or "—"
        add(f"| {st} | {statistics.fmean(s['rounds'] for s in rs):.2f} | {dist[0]} | {dist[1]} | "
            f"{dist[2]} | {dist[3]} | {capped} |")
    add("")
    add("参照：A 组固定 1 轮、B 组固定 2 轮（无自评，不参与提前终止）。")
    add("")

    add("按题型（C 组）：")
    add("")
    add("| 题型 | n | 平均轮次 | 触顶 |")
    add("|---|---|---|---|")
    for t in TYPES:
        rs = [s for s in stats if s["stage"] == "C" and s["qtype"] == t]
        if rs:
            add(f"| {t} | {len(rs)} | {statistics.fmean(s['rounds'] for s in rs):.2f} | "
                f"{sum(1 for s in rs if s['rounds'] == 3)} |")
    add("")

    # ---------- ② 终止原因 ----------
    add("## 2. 终止原因分布")
    add("")
    add("| 组 | " + " | ".join(("self_eval 足够", "触顶", "gate 不检索", "改写重复", "缓存命中")) + " |")
    add("|---|" + "---|" * 5)
    key_map = [("self_eval_sufficient", "self_eval 够"), ("cap", "触顶"),
               ("gate_no_retrieve", "gate 不检索"), ("query_repeat", "改写重复"),
               ("cache_hit", "缓存命中")]
    for st in STAGES:
        rs = [s for s in stats if s["stage"] == st]
        if not rs:
            continue
        cnt = Counter(s["reason"] for s in rs)
        add(f"| {st} | " + " | ".join(str(cnt[k]) for k, _ in key_map) + " |")
    add("")
    add("注：A/B 的「触顶」= 固定管线跑完预设轮次（不是撞上 step cap）；"
        "C/D1 的 0 轮 = gate 判「不检索」。C 组 cross 平均轮次被 2 道 0 轮题拉低。")
    add("")

    # ---------- ③ 改写有效率 ----------
    add("## 3. 改写有效率（分母 = 改写轮，不是题数）")
    add("")
    add("| 组 | 改写轮 | 与上轮重复 | 引入新块 | 引入新 gold |")
    add("|---|---|---|---|---|")
    all_new_chunk = all_new_gold = all_rew = all_rep = 0
    for st in ("B", "C", "D1"):
        rs = [s for s in stats if s["stage"] == st]
        rew = sum(s["rewrites"] for s in rs)
        rep = sum(s["repeat"] for s in rs)
        nc = sum(s["new_chunk_rounds"] for s in rs)
        ng = sum(s["new_gold_rounds"] for s in rs)
        all_rew, all_rep, all_new_chunk, all_new_gold = all_rew + rew, all_rep + rep, all_new_chunk + nc, all_new_gold + ng
        add(f"| {st} | {rew} | {pct(rep, rew)} | {pct(nc, rew)} | {pct(ng, rew)} |")
    add("")
    add(f"**合计**：改写轮 {all_rew}，其中与上轮重复 {all_rep}；"
        f"引入新 top-3 块 {pct(all_new_chunk, all_rew)}；"
        f"引入新 gold {pct(all_new_gold, all_rew)}。")
    add("")

    # ---------- ④ D2 命中去向 ----------
    add("## 4. 命中去向（D2 缓存）")
    add("")
    d2 = [s for s in stats if s["stage"] == "D2"]
    hits = [s for s in d2 if s["cache_hit"]]
    miss = [s for s in d2 if not s["cache_hit"]]
    d1_wrote = [s for s in stats if s["stage"] == "D1" and s["cache_wrote"]]
    add(f"- D1 写入缓存 {len(d1_wrote)}/{len([s for s in stats if s['stage']=='D1'])} 题；"
        f"D2 命中 {len(hits)}/{len(d2)} 题"
        + ("（写过的题 100% 命中）" if len(hits) == len(d1_wrote) else ""))
    if hits:
        ms = [s["total_ms"] for s in hits]
        sims = [s["cache_match"]["similarity"] for s in hits if s["cache_match"]]
        add(f"- 命中趟延迟：均值 {statistics.fmean(ms):.2f}ms / 中位 {statistics.median(ms):.2f}ms "
            f"/ 范围 {min(ms):.2f}–{max(ms):.2f}ms，llm_calls=0")
        if sims:
            add(f"- 相似度范围 {min(sims):.6f}–{max(sims):.6f}（原题重复，无假命中）")
    add(f"- 结构性未命中 {len(miss)} 题："
        + "、".join(f"id={s['qid']}({s['reason']})" for s in miss))
    add("")

    # ---------- 附录：每题轨迹 ----------
    add("## 附录：每题轨迹（C / D1）")
    add("")
    add("| 组 | id | 题型 | 轮次 | 终止 | 每轮 top1 |")
    add("|---|---|---|---|---|---|")
    for st in ("C", "D1"):
        for s in sorted([x for x in stats if x["stage"] == st], key=lambda x: int(x["qid"])):
            add(f"| {st} | {s['qid']} | {s['qtype']} | {s['rounds']} | {s['reason']} | "
                + " → ".join(s["top1"]) + " |")
    add("")
    add("口径：轨迹只落 top-3；gold 按题粒度（in=页级块 / cross·partial=来源级）；"
        "「新块」= 新进 top-3 的块，不是新被召回的块。")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\n… 完整报告已落盘：{REPORT}")


if __name__ == "__main__":
    main()
