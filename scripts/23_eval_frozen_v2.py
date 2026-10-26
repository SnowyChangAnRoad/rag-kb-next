"""W5 卡4 | frozen_v2 评测 harness（检索层零 API 基线 + 回归门禁）。

要回答的问题：30 道冻结题在【交付口径】的检索路径上表现如何，以及这个表现能不能
被 pytest 钉死成「不许退化」的门槛。

三层设计（卡5 的 pytest 直接复用本脚本，不再写第二套实现）：
    ① 指纹校验：frozen_v2.jsonl / chunks.jsonl 的 sha256 必须与 gold_frozen_v2.json
       记录的一致 —— gold 锚在旧题或旧语料上，指标会【静默失真】，绝不宽恕。
    ② 检索层（本脚本默认全跑，零 API）：
         召回（RECALL_K）→ 精排 → 去重，【不做配额与截断】
       口径与 W1 的 10_hybrid_ab.retrieve_stage 逐字一致：W1 通关口径是 recall@5，
       而 RERANK_TOP_K=3 —— 截断后再算，@5/@10 恒等于 @3，是假数；
       配额只决定「哪 3 条进 prompt」，属生成侧参数，不该污染召回段的结论。
    ③ 生成层（--answer，走 API）：本题不做。按 W5 决策，等卡3 把 judge 换成
       百炼 qwen-plus（不同家族 + 固定版本号）后一次性跑，避免 judge 同源问题
       重复花钱；本脚本先把检索层基线锁死。

口径要点（与 W1 相同，勿改）：
    · 外层题、内层 mode 交替 —— 先跑完 dense 再跑 hybrid 会让后跑的组系统性受益，
      方向恰好与假设同向，属于自欺型偏差；
    · warmup 一次（结果丢弃、不计时）—— 烧掉 CUDA 编译 / HNSW 载入 / BM25 缓存加载；
    · 延迟统计口径 = 全部【fact 题】（含 out 题），不按「有无 gold」筛。
      principle 题不检索（走 v1-noretr 出口），不计延迟 —— 否则测的是不存在的流量。
    · out 题（gold 为空）hits 照跑并落盘（它们也用于逐题回归），但不进打分表。

用法（项目根 rag-kb-next 下）：
    & "<venv>\\Scripts\\python.exe" scripts/23_eval_frozen_v2.py            # 跑基线并落盘
    & "<venv>\\Scripts\\python.exe" scripts/23_eval_frozen_v2.py --check    # 与基线严格比对（门禁）
    （基线已存在时默认拒绝覆盖 —— 改基线是一次显式决策，加 --force）

落盘：data/frozen_v2_baseline.json（含 sha256 指纹、config_fingerprint、逐题 hits、汇总指标）
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# scripts/ 不是包：运行前把项目根塞进 sys.path（同 10_hybrid_ab.py）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import (  # noqa: E402
    CHUNKS_JSONL, DEFAULT_EMBED, DEFAULT_VARIANT, FROZEN_V2_BASELINE_JSON,
    FROZEN_V2_JSONL, GOLD_FROZEN_V2_JSON, RECALL_K, RERANKER_MODEL_PATH,
    RERANK_TOP_K, RRF_K, SOURCE_CAP,
)
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.metrics import EVAL_KS, SOURCE_UNIT, evaluate  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

MODES = ("dense", "hybrid")


# ------------------------------------------------------------
# 数据装载：题目（kind/reference）+ gold（unit/gold）
# ------------------------------------------------------------
def load_records() -> list[dict]:
    """把 frozen_v2.jsonl 与 gold_frozen_v2.json 按 id 合并。

    为什么必须两边都读：
        gold 记录只有判分需要的字段（unit / gold / detail）；而 kind、reference
        留在题目文件里 —— gold 生成器不为判分之外的信息负责，也不该复制一份。
    失败必须显式：两边 id 集合不一致 = 冻结流程出了岔子（题目被改而 gold 没重跑），
    直接 raise，不静默取交集（取交集会让题目悄悄消失，分母变小、指标变好看）。
    """
    with open(FROZEN_V2_JSONL, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    payload = load_gold(GOLD_FROZEN_V2_JSON)     # 内部已校验 testset sha256
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
# 检索段（与 scripts/10_hybrid_ab.retrieve_stage 口径逐字一致）
# ------------------------------------------------------------
def retrieve_stage(collection, ranker, question: str, mode: str):
    """召回（RECALL_K）→ 精排 → 去重，返回 (hits, recall_ms, rerank_ms)。

    为什么不 import 10_hybrid_ab 的同名函数：scripts/ 不是包，W1 脚本已冻结，
    给冻结脚本加被依赖的接口 = 让历史的证据文件承担新职责。
    这 6 行的重复是刻意的；两处口径若将来要分叉，必须是有意识的决定。
    """
    t0 = time.perf_counter()
    hits = hybrid_retrieve(collection, question, RECALL_K, mode=mode)
    t1 = time.perf_counter()
    ranked = ranker.rerank(question, hits)
    deduped = dedupe_adjacent(ranked)
    t2 = time.perf_counter()
    return deduped, (t1 - t0) * 1000, (t2 - t1) * 1000


def latency_stats(pairs: list[tuple[float, float]]) -> dict:
    rc = [p[0] for p in pairs]
    rr = [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "recall_mean_ms": statistics.fmean(rc),
        "recall_med_ms": statistics.median(rc),
        "rerank_mean_ms": statistics.fmean(rr),
        "rerank_med_ms": statistics.median(rr),
        "total_med_ms": statistics.median([a + b for a, b in pairs]),
    }


def slim(hit: dict) -> dict:
    """落盘用的最小 hit：逐题回归只需要「顺序 + 身份 + 分数」。"""
    return {"chunk_id": hit["chunk_id"], "source": hit["source"],
            "page": hit.get("page"), "rerank_score": hit.get("rerank_score")}


def build_rows(collection, ranker, records: list[dict]):
    """跑全部 fact 题 × 两个 mode，返回 (rows_by_mode, lat_by_mode, hits_by_mode)。

    rows_by_mode 只放有 gold 的题（evaluate 的形状）；
    hits_by_mode 放全部 fact 题（含 out）—— 门禁要能抓到 out 题的检索漂移。
    """
    fact = [r for r in records if r["kind"] != "principle"]
    rows_by_mode: dict[str, list] = {m: [] for m in MODES}
    lat_by_mode: dict[str, list] = {m: [] for m in MODES}
    hits_by_mode: dict[str, list] = {m: [] for m in MODES}

    # warmup：每题两个 mode 各跑一次，结果丢弃、不计时、不入表
    for mode in MODES:
        retrieve_stage(collection, ranker, fact[0]["question"], mode)

    for r in fact:
        for mode in MODES:                       # 外层题、内层 mode（逐题交替）
            hits, t_recall, t_rerank = retrieve_stage(collection, ranker, r["question"], mode)
            lat_by_mode[mode].append((t_recall, t_rerank))
            hits_by_mode[mode].append({"id": r["id"], "type": r["type"],
                                       "hits": [slim(h) for h in hits]})
            if r["gold"]:                        # out 题无 gold → 不进打分表
                rows_by_mode[mode].append({"id": r["id"], "type": r["type"],
                                           "unit": r["unit"], "gold": r["gold"],
                                           "hits": hits})
            # 命中数要按该题的 gold 粒度数：source 级 gold 是「文件名集合」，
            # 直接拿 chunk_id 去比对会恒为 0（cross/partial 会全都显示 0/2，看不出真实命中）
            if r["unit"] == SOURCE_UNIT:
                n_hit = len({h["source"] for h in hits} & r["gold"])
            else:
                n_hit = sum(1 for h in hits if h["chunk_id"] in r["gold"])
            n_gold = len(r["gold"]) if r["gold"] else 0
            print(f"  {mode:<7} id={r['id']:<3} {r['type']:<8} "
                  f"命中 {n_hit}/{n_gold:<3} 召回 {t_recall:>6.1f}ms "
                  f"精排 {t_rerank:>6.1f}ms")

    return rows_by_mode, lat_by_mode, hits_by_mode


# ------------------------------------------------------------
# 打印
# ------------------------------------------------------------
def print_summary(metrics: dict, lat: dict, n_out: int) -> None:
    cols = [f"recall@{k}" for k in EVAL_KS] + ["mrr"] + [f"ndcg@{k}" for k in EVAL_KS]
    print(f"\n  {'组':<10}{'n':<4}" + "".join(f"{c:<11}" for c in cols))
    for group, m in metrics["by_type"].items():
        row = f"  {group:<10}{m['n_scored']:<4}"
        row += "".join(f"{m.get(c, float('nan')):<11.4f}" for c in cols)
        print(row)
    print(f"  {'混合(参考)':<10}{metrics['all_with_gold']['n_scored']:<4}"
          + "".join(f"{metrics['all_with_gold'].get(c, float('nan')):<11.4f}" for c in cols))
    print(f"  out 题 {n_out} 道：跑检索、计延迟，但无 gold，不进打分表")
    print(f"  延迟：召回均值 {lat['recall_mean_ms']:.1f}ms / 中位 {lat['recall_med_ms']:.1f}ms；"
          f"精排均值 {lat['rerank_mean_ms']:.1f}ms / 中位 {lat['rerank_med_ms']:.1f}ms；"
          f"合计中位 {lat['total_med_ms']:.1f}ms（{lat['n']} 题）")


# ------------------------------------------------------------
# 门禁比对（--check）
# ------------------------------------------------------------
def compare(baseline: dict, current: dict) -> list[str]:
    """严格比对：逐题 chunk_id 顺序 + rerank_score + 汇总指标。

    为什么逐题比而不是只比汇总：汇总可能互相抵消（一道题变好、另一道变差，均值不动），
    而那正是「改坏了却没被发现」的典型形态。
    """
    diffs: list[str] = []
    if baseline["config_fingerprint"] != current["config_fingerprint"]:
        diffs.append(f"config_fingerprint 变了：{baseline['config_fingerprint']} → "
                     f"{current['config_fingerprint']}")

    for mode in MODES:
        old, new = baseline["modes"][mode], current["modes"][mode]
        for group, old_m in old["metrics"]["by_type"].items():
            new_m = new["metrics"]["by_type"].get(group)
            if new_m is None:
                diffs.append(f"{mode}.{group}: 该题型消失了")
                continue
            for col, v in new_m.items():
                if old_m.get(col) != v:
                    diffs.append(f"{mode}.{group}.{col}: {old_m.get(col)!r} → {v!r}")

        old_rows = {r["id"]: r["hits"] for r in old["hits_all_fact"]}
        new_rows = {r["id"]: r["hits"] for r in new["hits_all_fact"]}
        if set(old_rows) != set(new_rows):
            diffs.append(f"{mode} 逐题集合变了：仅基线 {sorted(set(old_rows) - set(new_rows))} "
                         f"仅本次 {sorted(set(new_rows) - set(old_rows))}")
        for qid in sorted(set(old_rows) & set(new_rows)):
            a, b = old_rows[qid], new_rows[qid]
            if [h["chunk_id"] for h in a] != [h["chunk_id"] for h in b]:
                diffs.append(f"{mode} id={qid} top-k 顺序变了："
                             f"{[h['chunk_id'] for h in a]} → {[h['chunk_id'] for h in b]}")
                continue
            for i, (ha, hb) in enumerate(zip(a, b), 1):
                if ha["rerank_score"] != hb["rerank_score"]:
                    diffs.append(f"{mode} id={qid} #{i} {ha['chunk_id']} rerank_score "
                                 f"{ha['rerank_score']!r} → {hb['rerank_score']!r}")
    return diffs


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="与已有基线严格比对（门禁）")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的基线（显式升基线）")
    args = ap.parse_args()

    records = load_records()
    fact = [r for r in records if r["kind"] != "principle"]
    print(f"frozen_v2：{len(records)} 题 | fact {len(fact)} 题（含 out "
          f"{sum(1 for r in fact if not r['gold'])} 题）| principle "
          f"{len(records) - len(fact)} 题（不检索，跳过）")
    print(f"frozen_v2 sha256 = {sha256_file(FROZEN_V2_JSONL)}")

    collection = get_collection()
    ranker = BGEReranker()
    rows_by_mode, lat_by_mode, hits_by_mode = build_rows(collection, ranker, fact)

    fingerprint = {
        "modes": list(MODES), "recall_k": RECALL_K, "rerank_top_k": RERANK_TOP_K,
        "rrf_k": RRF_K, "source_cap": SOURCE_CAP, "dedupe": True,
        "min_score": 0.0, "default_embed": DEFAULT_EMBED, "variant": DEFAULT_VARIANT,
        "collection": collection.name, "eval_ks": list(EVAL_KS),
        # 库块数也进指纹：语料文件没变、但库被重建坏了（少块/多块）时，
        # 检索结果会变 → 门禁必须能指名是「库」还是「代码」出的问题
        "collection_count": collection.count(),
        "reranker": Path(RERANKER_MODEL_PATH).name,
    }

    current = {"modes": {}, "config_fingerprint": fingerprint}
    n_out = sum(1 for r in fact if not r["gold"])
    for mode in MODES:
        print(f"\n[{mode}] 汇总（{len(rows_by_mode[mode])} 题有 gold）")
        metrics = evaluate(rows_by_mode[mode], ks=EVAL_KS)
        lat = latency_stats(lat_by_mode[mode])
        print_summary(metrics, lat, n_out)
        current["modes"][mode] = {
            "metrics": metrics, "latency": lat,
            "n_fact": len(fact), "n_scored": len(rows_by_mode[mode]),
            # 只存这一份逐题 hits（含 out 题）：rows 与它是同一批数据的子集，
            # 两处都存会让基线里出现两份可独立漂移的「同一件事」
            "hits_all_fact": hits_by_mode[mode],
        }

    if args.check:
        if not FROZEN_V2_BASELINE_JSON.exists():
            raise SystemExit(f"基线不存在：{FROZEN_V2_BASELINE_JSON} —— 先跑一次（不带 --check）")
        with open(FROZEN_V2_BASELINE_JSON, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        if baseline["frozen_v2_sha256"] != sha256_file(FROZEN_V2_JSONL):
            raise SystemExit("基线锚定的 frozen_v2 已变（题目被改过）—— 门禁结论无意义")
        diffs = compare(baseline, current)
        if diffs:
            print(f"\n❌ 回归未通过（{len(diffs)} 处漂移）：")
            for d in diffs:
                print(f"  - {d}")
            raise SystemExit(1)
        print("\n✅ 回归通过：逐题 hits 顺序、rerank_score、汇总指标与基线【逐位相同】")
        return

    if FROZEN_V2_BASELINE_JSON.exists() and not args.force:
        raise SystemExit(f"{FROZEN_V2_BASELINE_JSON} 已存在 —— 基线不可覆盖"
                         f"（升基线 = 显式决策，加 --force 并说明理由）")

    baseline = {
        "version": "frozen_v2_baseline_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_v2": FROZEN_V2_JSONL.name,
        "frozen_v2_sha256": sha256_file(FROZEN_V2_JSONL),
        "chunks_sha256": sha256_file(CHUNKS_JSONL),
        "gold_version": "gold_frozen_v2",
        "config_fingerprint": fingerprint,
        "gate": {"tolerance": 0.0, "scope": "逐题 chunk_id 顺序 + rerank_score + 汇总指标",
                 "node": "检索层零 API；生成层不入默认门禁"},
        "modes": current["modes"],
    }
    with open(FROZEN_V2_BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"\n基线已落盘：{FROZEN_V2_BASELINE_JSON}")
    print(f"  sha256(frozen_v2) = {baseline['frozen_v2_sha256']}")
    print("⚠️ 再加跑一遍确认可复现：本脚本 --check 应与刚落的基线逐位相同")


if __name__ == "__main__":
    main()
