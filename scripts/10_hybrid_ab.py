"""实验 | W1 卡7：dense-only vs dense+BM25(RRF) 首张对照表。

要回答的问题（W1 通关口径）：在同一份 tune 集（20 题）上，把召回段从
「纯向量」换成「向量 ⊕ BM25 → RRF」，检索层指标有没有真的变好，代价是多少。

一张 A/B 表要成立，难的不是画表，是下面四件事：

1) 单一实验变量
   hybrid_retrieve 的 mode 参数存在的唯一目的：让两组【共用除召回段以外的全部代码路径】。
   所以本脚本里【没有一处 if mode / if 实验组】—— 唯一的差异是传给
   hybrid_retrieve(..., mode=mode) 的那一个实参。脚本自己若分叉成两条调用链，
   将来改一处忘另一处，对照表会静默失真（fusion.py 文件头第 2 条硬约束）。

2) 指标算在哪一层
   本表算在【精排 + 去重后、配额截断前】的全量降序列表上（实测约 7~10 条）：
       - W1 通关口径是 recall@5，而 RERANK_TOP_K=3：截断后再算，@5/@10 恒等于 @3，是假数
       - 配额 + 截断只决定「哪 3 条进 prompt」，属生成层；把它算进检索层指标，
         等于让一个生成侧参数污染召回段的结论
   去重【保留在评测路径里】，因为它是 Day9 定下的、两组都开着的固定项。

3) 热态偏置
   第一次调用里混着 CUDA kernel 编译 / HNSW 首次载入 / BM25 token 缓存加载，
   第一组的 recall_ms 会明显偏高 —— 看起来像「hybrid 更慢」，其实是「它第二个跑，已经热了」。
   所以：先 warmup（不计时），再【逐题交替】跑两个 mode（外层题、内层 mode），
   让两组共享同一 GPU 热态与页面缓存。
   ⚠️ 绝不能写成「先跑完 dense 20 题，再跑完 hybrid 20 题」：后跑的那组系统性受益，
      而方向恰好与本实验的假设同向 —— 这是自欺型偏差。

4) 确定性
   temperature=0 + RRF 的 (-rrf, chunk_id) tie-break + metrics 不重排，
   三者合起来应当保证【同一份数据重跑逐位同分】。跑完顺手重跑一遍 dense 组断言这一点：
   若同一组自己都会漂，那「hybrid 高了 X pt」里的 X 就根本不可信。

用法（项目根 rag-kb-next 下）：
    & "<项目 venv>\\Scripts\\python.exe" scripts/10_hybrid_ab.py
    # 需要 GPU（精排）+ 已建好的 day6_kb（932 块）+ 已生成 data/gold_tune.json
    # 跑完落盘 data/eval_w1_ab.json（含逐题明细，卡8 直接读，不重跑 GPU）
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包，运行时把项目根塞进 sys.path 才能 import rag.*（同 05_split.py 的做法）
sys.path.insert(0, str(ROOT))

from rag.config import (  # noqa: E402
    GOLD_TUNE_JSON, RECALL_K, RERANK_TOP_K, RRF_K, SOURCE_CAP, TESTSET_TUNE_JSONL,
)
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.metrics import EVAL_KS, evaluate  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

MODES = ("dense", "hybrid")
EVIDENCE_JSON = ROOT / "data" / "eval_w1_ab.json"
COLS = ["recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10"]
GROUP_LABEL = {
    "in": "in 题（页级 gold）",
    "cross": "cross 题（来源级 gold）",
    "partial": "partial 题（来源级 gold）",
    "all_with_gold": "全题混合（⚠️ 粒度和题型都不同，仅供参照，别当 headline）",
}


# ------------------------------------------------------------
# 检索段（本脚本唯一分叉的地方就是一个实参）
# ------------------------------------------------------------
def retrieve_stage(collection, ranker, question: str, mode: str):
    """TODO-1：跑「召回 → 精排 → 去重」，返回 (hits, recall_ms, rerank_ms)。

    ⚠️ 顺序不能错，也不能少一步 —— 这就是 pipeline._retrieve 的 4 步里的前 3 步
    （第 4 步「配额 + 截断」被本表的指标层级刻意排除，理由见文件头第 2 条）。

    要求：
      a) 召回：hybrid_retrieve(collection, question, RECALL_K, mode=mode)
         —— 【不要】在这里写 if mode 分支。基线组也走同一个函数，只是 mode="dense"，
            它内部会退化成 retrieve()（fusion.py 已给的分支）。
            你自己分叉 = 两条调用链 = 将来改一处忘另一处 = 对照表静默失真
      b) 精排：ranker.rerank(question, hits) —— 返回【全量降序】，方法内部不截断
      c) 去重：dedupe_adjacent(ranked) —— 两组都开，是固定项
      d) 计时只测【这两段】，用 time.perf_counter()：
            t0 → t1 = 召回（含 query 编码，走 GPU）
            t1 → t2 = 精排 + 去重
         为什么要拆成两段而不是只测总时长：
            召回的差异（多跑了一路 BM25 + 分词）和精排的差异（两组候选池长度不同）
            是两笔不同的账。合成一个数就没法回答「hybrid 慢的那 X ms 花在哪」。
         ⚠️ 第一次调用含 CUDA 编译等一次性成本 —— warmup 由 build_rows 负责，
            这里不要自作聪明地「第一次不计时」（那会让本函数行为依赖调用顺序）

    返回：pipeline._retrieve 第 3 步结束时的降序列表（未配额、未截断）。
    """
    t0 = time.perf_counter()
    hits = hybrid_retrieve(collection, question, RECALL_K, mode=mode)
    t1 = time.perf_counter()
    ranked = ranker.rerank(question, hits)
    deduped = dedupe_adjacent(ranked)
    t2 = time.perf_counter()
    # 返回【毫秒】而不是秒：下游 latency_stats 的列名就是 _ms，统一单位免得到处换算
    return deduped, (t1 - t0) * 1000, (t2 - t1) * 1000


# ------------------------------------------------------------
# 主循环：warmup + 逐题交替
# ------------------------------------------------------------
def build_rows(collection, ranker, records: list[dict]):
    """TODO-2：跑完两组，返回 (rows_by_mode, lat_by_mode)。

    rows_by_mode[mode] 必须是 metrics.evaluate 要的形状：
        [{"id", "type", "unit", "gold": set[str], "hits": [...]}, ...]
    gold / unit 直接来自 records 里同名的键（load_gold 已把 gold 转成 set）。

    records 长什么样（就是 gold_tune.json 落盘的形状，main 里传的是 load_gold()["records"]）：
        {"id": 1, "type": "in",
         "question": "RAG 系统根据提出动因和特点可以分为哪几类？…",  ← 题目文本在这里，不用额外传参
         "expect": "RAG知识.pdf 第16页",
         "unit": "page",
         "gold": {"RAG知识_p016_c001", ...},                       ← load_gold 已把 list 转成 set
         "detail": {"files": ["RAG知识.pdf"], "page": 16, "n_chunks": 4}}
    ⚠️ "records"（复数）是 gold_tune.json 的顶层键，记录本身没有叫 record 的键 ——
       每条记录就是一个扁平 dict，上面这 7 个键。取题目文本用 r["question"]。

    lat_by_mode[mode] = [(recall_ms, rerank_ms), ...]，供 latency_stats 用。

    要写对的四件事：
      a) warmup：先拿 records[0]，两个 mode 各跑一次，【结果丢弃、不计时、不入表】。
         目的只是烧掉 CUDA kernel 编译 / HNSW 首次载入 / BM25 token 缓存加载。
      b) 逐题交替：外层 for 题、内层 for mode。同一道题连着跑 dense 和 hybrid。
         写成外层 mode、内层题的话，后跑的那组系统性受益 —— 方向恰好对本实验假设有利，
         属于自己骗自己（文件头第 3 条）。
      c) gold 为空的题（out 题）：hits 照跑（延迟要统计，它们是真实流量），
         但【不进 rows】—— evaluate 反正会跳过，提前剔除能让"参与打分的题数"一目了然。
         ⚠️ 延迟统计的口径是【全部 20 题、两组同源】，不要按「有 gold 的题」筛 ——
            否则两组题集不同，延迟又变成了第二个变量。
      d) 打印进度：20 题 × 2 组，跑到一半没输出会让人以为卡死。
         每题一行，带上 mode / 命中位数 / 两段耗时（如
         "dense   id=16 recalls=3  recall=182.1ms rerank=94.3ms"）。

    返回：(rows_by_mode, lat_by_mode)
    """
    rows_by_mode: dict[str, list] = {m: [] for m in MODES}   # 必须按 mode 分组：
    lat_by_mode: dict[str, list] = {m: [] for m in MODES}    # evaluate 要的是「一组的 rows」

    # warmup：两个 mode 各跑一次，结果丢弃、不计时、不入表
    for mode in MODES:
        retrieve_stage(collection, ranker, records[0]["question"], mode)

    for r in records:
        for mode in MODES:                    # 外层题、内层 mode（逐题交替）
            hits, t_recall, t_rerank = retrieve_stage(collection, ranker, r["question"], mode)
            # 延迟口径＝全部 20 题、两组同源。不按「有没有 gold」筛，
            # 否则两组题集不同，延迟又变成第二个变量
            lat_by_mode[mode].append((t_recall, t_rerank))
            if r["gold"]:                     # out 题没有 gold → 不进打分表（evaluate 也会跳过）
                rows_by_mode[mode].append({**r, "hits": hits})
            # 命中位数要自己数：r["detail"]["n_chunks"] 是【gold 的总块数】，不是命中数
            n_hit = sum(1 for h in hits if h["chunk_id"] in r["gold"])
            print(f"{mode:<7} id={r['id']:<3} 命中 {n_hit}/{len(r['gold'])}  "
                  f"召回 {t_recall:.1f}ms  精排+去重 {t_rerank:.1f}ms")

    return rows_by_mode, lat_by_mode


# ------------------------------------------------------------
# 确定性自检（免费的「数字可信」证据）
# ------------------------------------------------------------
def determinism_check(collection, ranker, records: list[dict],
                      before: dict, verbose: bool = False) -> None:
    """TODO-3：重跑一遍 dense 组，断言指标【逐位相同】。

    为什么值得多花这 2 秒：W1 的结论是「hybrid 的 recall@5 比 dense 高 X pt」。
    若同一份数据重跑一次 dense 组就会漂，那 X 的【量级】必须重新评估 ——
    这不是严谨性洁癖，Day9 已经踩过一次：temperature=0.7 时两组 top3 逐字相同、
    指标却不同，采样噪声直接盖过实验变量。

    要求：
      1) 重新跑一遍 mode="dense"（复用 build_rows 的逻辑，或直接调 retrieve_stage），
         得到新的 rows，再 evaluate 一次
      2) 与传进来的 before（dense 组的 summary）逐字段比较：
         比 by_type 下每个题型的 recall@3/@5/@10 / ndcg@3/@5/@10 / mrr
      3) 相同 → 打印一行通过（说明「温度锁 0 + tie-break + 指标不重排」这条链真的成立）
         不同 → 打印【哪些字段、差多少】再 raise。
         ⚠️ 别只 raise 一句「不一致」—— 下次你会花一小时找是哪一项漂了
      4) verbose=True 时把两遍的数字都打出来（默认关，因为这行会被反复看到）

    提示：比较用 != 而不是 isclose —— 这条链的目标是【逐位相同】，
    isclose 会把「小漂移」放过去，而漂移正是要抓的东西。
    """
    # 重跑 dense 组：只跑这一组（跑两组是浪费），重建 rows 再 evaluate
    rows = [{**r, "hits": retrieve_stage(collection, ranker, r["question"], "dense")[0]}
            for r in records if r["gold"]]
    after = evaluate(rows)          # ← evaluate 的第二个参数是 ks（评估深度），不是延迟数据

    if verbose:
        print("重跑前：", json.dumps(before, ensure_ascii=False))
        print("重跑后：", json.dumps(after, ensure_ascii=False))

    diffs: list[str] = []
    for group, new_g in after["by_type"].items():  # 外层：分组（in / cross / partial）
        old_g = before["by_type"].get(group)  # ← 用 .get 不用 []
        if old_g is None:
            diffs.append(f"by_type.{group}: 重跑后多出/少了这个题型")
            continue
        for col, v_new in new_g.items():  # 内层：字段（mrr / recall@3 / …）
            v_old = old_g.get(col)
            if v_old != v_new:  # ← != 而不是 isclose
                diffs.append(f"{group}.{col}: {v_old!r} → {v_new!r}")
            elif verbose:
                print(f"  {group}.{col}: {v_new!r}  （相同）")

    if diffs:
        print("确定性自检未通过：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"dense 组重跑后指标漂移，共 {len(diffs)} 个字段")
    print("确定性自检通过：dense 组重跑后指标【逐位相同】")
    

# ------------------------------------------------------------
# 以下为已给：延迟统计 / 打印 / 落盘
# ------------------------------------------------------------
def latency_stats(pairs: list[tuple[float, float]]) -> dict:
    """已给：两段耗时的均值与中位数。

    为什么必须带中位数：20 题里偶发一次 CUDA 同步或 GC 就能把均值拉高几十 ms，
    而中位数对单次抖动免疫。也别只报中位数 —— 两者的差就是抖动幅度的信号。
    """
    rc = [p[0] for p in pairs]
    rr = [p[1] for p in pairs]
    return {
        "recall_mean_ms": statistics.fmean(rc), "recall_med_ms": statistics.median(rc),
        "rerank_mean_ms": statistics.fmean(rr), "rerank_med_ms": statistics.median(rr),
        "total_med_ms": statistics.median([a + b for a, b in pairs]),
    }


def _pick(summary: dict, group: str) -> dict:
    return summary["all_with_gold"] if group == "all_with_gold" else summary["by_type"].get(group)


def print_table(summaries: dict, lat: dict) -> None:
    """已给：指标对照表（每个题型 = 一组，因为粒度不同不可混）。"""
    print(f"\n参数：RECALL_K={RECALL_K} RRF_K={RRF_K} RERANK_TOP_K={RERANK_TOP_K}"
          f"（本表【未截断】）SOURCE_CAP={SOURCE_CAP}")
    print(f"指标层级：精排 + 去重后全量 | 深度：@3/@5/@10 | 执行：warmup + 逐题交替")

    for group in ("in", "cross", "partial", "all_with_gold"):
        base = _pick(summaries[MODES[0]], group)
        if not base:
            continue
        print(f"\n--- {GROUP_LABEL[group]} ---")
        print(f"{'组别':<20}{'n':>3} " + "".join(f"{c:>10}" for c in COLS))
        for m in MODES:
            g = _pick(summaries[m], group)
            cells = "".join(f"{g.get(c, float('nan')):>10.4f}" for c in COLS)
            print(f"{m + '-only' if m == 'dense' else m + ' (RRF)':<20}{g['n_scored']:>3} {cells}")
        delta = "".join(
            f"{_pick(summaries['hybrid'], group).get(c, 0) - base.get(c, 0):>+10.4f}" for c in COLS
        )
        print(f"{'Δ (hybrid - dense)':<20}{'':>3} {delta}")

    print("\n--- 延迟（全部 20 题，两组同源）---")
    print(f"{'组别':<20}{'召回均值':>10}{'召回中位':>10}{'精排均值':>10}{'精排中位':>10}{'合计中位':>10}")
    for m in MODES:
        l = lat[m]
        print(f"{m + '-only' if m == 'dense' else m + ' (RRF)':<20}"
              f"{l['recall_mean_ms']:>10.1f}{l['recall_med_ms']:>10.1f}"
              f"{l['rerank_mean_ms']:>10.1f}{l['rerank_med_ms']:>10.1f}{l['total_med_ms']:>10.1f}")
    dl = lat["hybrid"]
    db = lat["dense"]
    print(f"{'Δ (hybrid - dense)':<20}"
          f"{dl['recall_mean_ms'] - db['recall_mean_ms']:>+10.1f}"
          f"{dl['recall_med_ms'] - db['recall_med_ms']:>+10.1f}"
          f"{dl['rerank_mean_ms'] - db['rerank_mean_ms']:>+10.1f}"
          f"{dl['rerank_med_ms'] - db['rerank_med_ms']:>+10.1f}"
          f"{dl['total_med_ms'] - db['total_med_ms']:>+10.1f}")


def dump_evidence(summaries: dict, lat: dict, rows_by_mode: dict) -> None:
    """已给：落盘。含指纹（可溯源）+ 逐题明细（卡8 免重跑 GPU）。

    为什么带指纹：W5 对外报数时要能回答「当时跑的是哪份 gold、哪份测试集」。
    gold 自身已经钉了 testset_sha256，这里再钉一层 gold_sha256，链条才闭合。
    """
    per_question = {}
    for m, rows in rows_by_mode.items():
        per_question[m] = [
            {
                "id": r["id"], "type": r["type"], "unit": r["unit"],
                "n_gold": len(r["gold"]),
                "hit_ranks": [i for i, h in enumerate(r["hits"], 1)
                              if h["chunk_id"] in r["gold"]],
                "hits": [
                    {"rank": i, "chunk_id": h["chunk_id"], "rerank_score": round(h.get("rerank_score", 0.0), 4),
                     **({"source": h["source"]} if "source" in h else {}),
                     **({"page": h["page"]} if "page" in h else {}),
                     **({"rrf": round(h["rrf"], 6)} if "rrf" in h else {}),
                     "in_gold": h["chunk_id"] in r["gold"],
                     "preview": h["text"][:80]}
                    for i, h in enumerate(r["hits"], 1)
                ],
            }
            for r in rows
        ]

    payload = {
        "experiment": "w1_dense_vs_hybrid",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "RECALL_K": RECALL_K, "RRF_K": RRF_K, "RERANK_TOP_K": RERANK_TOP_K,
            "SOURCE_CAP": SOURCE_CAP, "ks": list(EVAL_KS),
            "metric_layer": "post_dedupe_pre_quota",
        },
        "fingerprints": {
            "testset": TESTSET_TUNE_JSONL.name,
            "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
            "gold": GOLD_TUNE_JSON.name,
            "gold_sha256": sha256_file(GOLD_TUNE_JSON),
        },
        "metrics": summaries,
        "latency": lat,
        "per_question": per_question,
    }
    with open(EVIDENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{EVIDENCE_JSON}")


def main() -> None:
    """已给：装配 → 跑两组 → 指标 → 确定性自检 → 打印 → 落盘。"""
    records = load_gold()["records"]        # gold 已是 set；unit 也在记录里
    collection = get_collection()
    ranker = BGEReranker()                  # 只加载一次，两组共用（省显存 + 保证环境一致）

    rows_by_mode, lat_by_mode = build_rows(collection, ranker, records)
    summaries = {m: evaluate(rows_by_mode[m]) for m in MODES}
    lat = {m: latency_stats(lat_by_mode[m]) for m in MODES}

    print_table(summaries, lat)

    determinism_check(collection, ranker, records, summaries["dense"])
    dump_evidence(summaries, lat, rows_by_mode)


if __name__ == "__main__":
    main()