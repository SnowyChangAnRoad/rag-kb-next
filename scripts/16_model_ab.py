"""实验 | W2 卡4：embedding × reranker 2×2 交叉对照。

要回答的问题
    把 W1 的检索层（dense 召回 → 精排 → 去重）原封搬到两个新模型上，
    质量涨多少、延迟 / 显存涨多少，拐点在哪。

实验纪律（沿用 W1 卡7 / scripts/10_hybrid_ab.py，一项都不放松）
    1) 单一变量：4 格只有 (embed_key, reranker) 不同，其余共用【同一条代码路径】。
       ⚠️ 本脚本【不许出现 if cell 分支】——两条调用链 = 将来改一处忘另一处 = 对照表静默失真。
    2) 指标层：post-dedupe pre-quota（精排 + 去重后、配额截断前的全量降序列表）。
       配额 + 截断属生成层，算进来会让生成侧参数污染召回段结论。
    3) 热态：warmup 先行，再【外层题 / 内层格】交替。绝不"一格跑完再跑下一格"——
       后跑的格系统性受益，方向又恰好对本实验假设有利，属自欺型偏差。
    4) 确定性：temperature=0 + 指标不重排 → 重跑基线格应逐位同分。

⚠️ 前提（卡4 的接线改动，先做完再跑）
    fusion.hybrid_retrieve 已新增 model_key 参数，并【透传到它内部两处 retrieve() 调用】
    （mode=="dense" 分支 + TODO-1 那行）。少透传一处，qwen3 两格会按 512 维编码 → 维度报错。

用法（项目根 rag-kb-next 下）
    & "<项目 venv>\\Scripts\\python.exe" scripts/16_model_ab.py
    # 需要 GPU；qwen3_kb 已由 scripts/14_embed_check.py 建好；data/gold_tune.json 已存在
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包，运行时把项目根塞进 sys.path 才能 import rag.*（同 10_hybrid_ab.py）
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rag.config import (  # noqa: E402
    GOLD_TUNE_JSON, RECALL_K, RERANKER_MODEL_PATH, RERANK_TOP_K, SOURCE_CAP,
    TESTSET_TUNE_JSONL,
)
from rag.embedding import get_embedder  # noqa: E402
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.metrics import evaluate  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

# ------------------------------------------------------------
# 对照轴：4 格，唯一不同的就是这两个字段
# ------------------------------------------------------------
# reranker 不进注册表（它不建索引、不入 collection，无跨文件的硬耦合）→ 路径在脚本里显式给。
# 基线用 config.RERANKER_MODEL_PATH，保证"运行时默认 reranker"仍只有一处事实。
V2M3_RERANKER = (r"c:\Users\xx\Desktop\python\Agent学习\PyTorch\HuggingFace\models"
                 r"\bge-reranker-v2-m3")
GRID = {
    "base__base":  {"embed_key": "bge-small-zh", "reranker_path": RERANKER_MODEL_PATH},
    "qwen3__base": {"embed_key": "qwen3-0.6b",   "reranker_path": RERANKER_MODEL_PATH},
    "base__v2m3":  {"embed_key": "bge-small-zh", "reranker_path": V2M3_RERANKER},
    "qwen3__v2m3": {"embed_key": "qwen3-0.6b",   "reranker_path": V2M3_RERANKER},
}
BASELINE = "base__base"          # 回归闸：基线格指标必须 == W1 的 dense 组
W1_DENSE_JSON = ROOT / "data" / "eval_w1_ab.json"
EVIDENCE_JSON = ROOT / "data" / "eval_w2_model_ab.json"
COLS = ["recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10"]
GROUP_LABEL = {
    "in": "in 题（页级 gold）",
    "cross": "cross 题（来源级 gold）",
    "partial": "partial 题（来源级 gold）",
    "all_with_gold": "全题混合（⚠️ 粒度/题型都不同，仅供参照）",
}


# ------------------------------------------------------------
# 装配（已给）
# ------------------------------------------------------------
_RANKER_CACHE: dict[str, BGEReranker] = {}


def _ranker(path: str) -> BGEReranker:
    """已给：按路径缓存精排器 —— 两个 base 格指向同一 path，别把同一份权重建两次。"""
    if path not in _RANKER_CACHE:
        _RANKER_CACHE[path] = BGEReranker(path)
    return _RANKER_CACHE[path]


def build_runners() -> dict:
    """已给：按 GRID 装配 4 格的 (collection, ranker)。

    embedding 侧走 get_embedder 的缓存（embedding.py 内），精排侧走上面的 _ranker 缓存。
    """
    runners = {}
    for label, cell in GRID.items():
        runners[label] = (get_collection(model_key=cell["embed_key"]),
                          _ranker(cell["reranker_path"]))
    return runners


# ------------------------------------------------------------
# 检索段（本脚本唯一分叉的地方就是两个实参）
# ------------------------------------------------------------
def retrieve_stage(collection, ranker, question: str, model_key: str):
    """TODO-1：跑「召回 → 精排 → 去重」，返回 (hits, recall_ms, rerank_ms)。

    与 W1 的 retrieve_stage 同形，只多一个 model_key 透传。契约：
      输入  collection/ranker：这一格的库与精排器
            question         ：题目文本
            model_key        ：这一格的嵌入模型 key（决定 query 用哪套编码 / 多少维）
      输出  (hits, recall_ms, rerank_ms)
            hits      = 去重后的【全量降序】列表（未配额、未截断），供评分
            recall_ms = 召回段耗时（含 query 编码，走 GPU）
            rerank_ms = 精排 + 去重耗时
      约束  · 不在这里写 if cell —— 4 格走同一行代码，只有实参不同
            · 计时只覆盖「召回」与「精排+去重」两段
            · 第一次调用的一次性成本（CUDA 编译等）由 build_rows 的 warmup 负责，
              这里不要特判"第一次不计时"（那会让行为依赖调用顺序）
    """
    t0 = time.perf_counter()
    # ⚠️ mode / model_key 必须用【关键字】传：hybrid_retrieve 第 4 个位置参数是 mode，不是 model_key。
    #    写成位置传参会把 embed_key 当 mode → ValueError（响得早，但方向全错）。
    hits = hybrid_retrieve(collection, question, RECALL_K, mode="dense", model_key=model_key)
    t1 = time.perf_counter()
    ranked = ranker.rerank(question, hits)
    deduped = dedupe_adjacent(ranked)
    t2 = time.perf_counter()
    return deduped, (t1 - t0) * 1000, (t2 - t1) * 1000


def hit_key(h: dict, unit: str):
    """已给：命中判据按 gold 口径分流（卡5 回补）。

    ⚠️ cross/partial 是【来源级】gold，拿 chunk_id 去比永远不命中 ——
       指标层没事（evaluate 按 unit 分流），但进度行与证据文件会全是 0，属误导性空值。
    """
    return h["chunk_id"] if unit == "page" else h.get("source")


# ------------------------------------------------------------
# 主循环：warmup + 逐题交替
# ------------------------------------------------------------
def build_rows(runners: dict, records: list[dict]):
    """TODO-2：跑完 4 格，返回 (rows_by_cell, lat_by_cell)。

    契约：
      rows_by_cell[cell] 必须是 metrics.evaluate 要的形状：
          [{"id", "type", "unit", "gold", "hits": [...]}, ...]
          ← 由 records 里的同名字段 + 本格算出的 hits 组成（{**r, "hits": hits}）
      lat_by_cell[cell] = [(recall_ms, rerank_ms), ...]，供 latency_stats 用
      records 形状见 gold_tune.json：题目文本在 r["question"]，r["gold"] 已是 set

    硬性要求（与 W1 build_rows 一一对应，任何一条都不能省）：
      a) warmup：拿 records[0]，4 格各跑一次，结果丢弃 / 不计时 / 不入表
         （烧掉 CUDA 编译 / HNSW 首次载入等一次性成本）
      b) 逐题交替：外层 for 题、内层 for 格。同一道题连着跑完 4 格
         （写成外层格、内层题 → 后跑的格系统性受益，方向对本实验假设有利）
      c) gold 为空的题（out 题）：hits 照跑（延迟要统计，是真实流量），但不进 rows
         （evaluate 会跳过；提前剔除能让"参与打分的题数"一目了然）
         ⚠️ 延迟口径 = 全部 20 题、4 格同源；不要按"有 gold 的题"筛
      d) 每题每格打一行进度（20×4 无输出会让人以为卡死）
    """
    rows_by_mode: dict[str, list] = {g: [] for g in GRID}
    lat_by_mode: dict[str, list] = {g: [] for g in GRID}

    # Warm_up 每个组先跑一次record[0]
    # ⚠️ 第 4 个实参是【embed_key】（GRID[g]["embed_key"]），不是循环变量 g（那是 GRID 标签）。
    for g in GRID:
        retrieve_stage(runners[g][0], runners[g][1], records[0]["question"], GRID[g]["embed_key"])

    for r in records:
        for g in GRID:
            hits, t_recall, t_rerank = retrieve_stage(
                runners[g][0], runners[g][1], r["question"], GRID[g]["embed_key"])
            lat_by_mode[g].append((t_recall, t_rerank))
            if r["gold"]:
                rows_by_mode[g].append({**r, "hits": hits})

            n_hit = sum(1 for h in hits if hit_key(h, r["unit"]) in r["gold"])
            print(f"{g:<10} id={r['id']:<3} 命中 {n_hit}/{len(r['gold'])}  "
                  f"召回 {t_recall:.1f}ms  精排+去重 {t_rerank:.1f}ms")

    return rows_by_mode, lat_by_mode


# ------------------------------------------------------------
# 自检：确定性 + 基线回归（两道闸）
# ------------------------------------------------------------
def determinism_check(runners: dict, records: list[dict], before: dict) -> None:
    """TODO-3：重跑基线格，断言指标【逐位相同】。

    契约：
      输入  before = 首轮基线格的 summary（evaluate 的输出）
      要求  重跑基线格 → 再 evaluate → 与 before 逐字段比较
            （by_type 下每个题型的 recall@3/@5/@10 / ndcg@3/@5/@10 / mrr）
      相同 → 打印一行通过
      不同 → 打印【哪些字段、差多少】再 raise（别只 raise 一句"不一致"）
      ⚠️ 比较用 != 而不是 isclose —— 目标是逐位相同，isclose 会把小漂移放过去
    """
    key = BASELINE
    # ⚠️ retrieve_stage 返回【三元组】，取 [0] 才是 hits；直接把元组塞进 "hits" 会在 evaluate 里炸。
    rows = [{**r, "hits": retrieve_stage(runners[key][0], runners[key][1],
                                        r["question"], GRID[key]["embed_key"])[0]}
            for r in records if r["gold"]]
    after = evaluate(rows)  # ← evaluate 的第二个参数是 ks（评估深度），不是延迟数据

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

    if diffs:
        print("确定性自检未通过：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"基线组重跑后指标漂移，共 {len(diffs)} 个字段")
    print("确定性自检通过：基线组重跑后指标【逐位相同】")


def baseline_regression_check(base_summary: dict) -> None:
    """TODO-4：断言基线格指标 == W1 的 dense 组（卡4 接线改动的回归闸）。

    为什么单列这一道：卡4 动了两处接线（hybrid_retrieve 透传 model_key）。
    若基线格对不上 W1 的 dense 组，说明接线引入了偏差 —— 那"qwen3 涨了 X pt"就不干净。

    契约：
      读 W1_DENSE_JSON（10_hybrid_ab.py 落盘），结构里 metrics["dense"] 与本模块
      evaluate 的输出【同构】（by_type / all_with_gold / n_no_gold）。
      与 base_summary 逐字段 != 比较 by_type 下每个题型的全部指标列；
      一致 → 打印通过；不一致 → 列出差异字段再 raise。
    """
    # W1 落盘结构：payload["metrics"] = {"dense": <evaluate 输出>, "hybrid": <evaluate 输出>}
    # 所以要的是 payload["metrics"]["dense"]，它就是 evaluate(rows) 的返回值，形状与你手上的
    # base_summary 完全同构：{"by_type": {"in": {...}, "cross": {...}, ...},
    #                          "all_with_gold": {...}, "n_no_gold": int}
    # 每个题型下又是 {"n", "n_scored", "mrr", "recall@3/5/10", "ndcg@3/5/10"}。
    with open(W1_DENSE_JSON, "r", encoding="utf-8") as f:
        w1_dense = json.load(f)["metrics"]["dense"]

    diffs: list[str] = []
    for group, new_g in base_summary["by_type"].items():   # 外层：题型（in / cross / partial）
        old_g = w1_dense["by_type"].get(group)
        if old_g is None:
            diffs.append(f"by_type.{group}: W1 里没有这个题型")
            continue
        for col, v_new in new_g.items():                   # 内层：字段（mrr / recall@3 / …）
            v_old = old_g.get(col)
            if v_old != v_new:                             # ← != 而不是 isclose
                diffs.append(f"{group}.{col}: W1={v_old!r} → 本次={v_new!r}")

    if diffs:
        print("基线回归未通过（接线改动引入了偏差）：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"基线格 != W1 dense 组，共 {len(diffs)} 个字段")
    print("基线回归通过：基线格 == W1 dense 组（接线改动零偏差）")


# ------------------------------------------------------------
# 以下为已给：延迟统计 / 打印 / 落盘
# ------------------------------------------------------------
def report_footprint(runners: dict) -> None:
    """已给：全模型加载后的显存 + 各格两模型的参数量（回答"显存占用"那一列）。"""
    print(f"显存：4 格全部加载后 allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")
    for label, cell in GRID.items():
        emb = get_embedder(cell["embed_key"]).model
        rk = runners[label][1].model
        print(f"  {label:<12} embed={sum(p.numel() for p in emb.parameters()) / 1e6:>5.0f}M "
              f"rerank={sum(p.numel() for p in rk.parameters()) / 1e6:>5.0f}M")


def latency_stats(pairs: list[tuple[float, float]]) -> dict:
    """已给：两段耗时的均值与中位数（中位数对单次 CUDA 抖动免疫，差值 = 抖动幅度信号）。"""
    rc = [p[0] for p in pairs]
    rr = [p[1] for p in pairs]
    return {
        "recall_mean_ms": statistics.fmean(rc), "recall_med_ms": statistics.median(rc),
        "rerank_mean_ms": statistics.fmean(rr), "rerank_med_ms": statistics.median(rr),
        "total_med_ms": statistics.median([a + b for a, b in pairs]),
    }


def _pick(summary: dict, group: str) -> dict:
    return summary["all_with_gold"] if group == "all_with_gold" else summary["by_type"].get(group)


def print_table(metrics: dict, lat: dict) -> None:
    """已给：指标对照表（每个题型一组）+ 延迟表。"""
    print(f"\n参数：RECALL_K={RECALL_K} RERANK_TOP_K={RERANK_TOP_K}"
          f"（本表【未截断】）SOURCE_CAP={SOURCE_CAP} mode=dense")
    print(f"指标层级：精排 + 去重后全量 | 深度：@3/@5/@10 | 执行：warmup + 逐题交替")
    print(f"对照轴：embedding × reranker | 基线格 = {BASELINE}")

    for group in ("in", "cross", "partial", "all_with_gold"):
        base = _pick(metrics[BASELINE], group)
        if not base:
            continue
        print(f"\n--- {GROUP_LABEL[group]} ---")
        print(f"{'格':<20}{'n':>3} " + "".join(f"{c:>10}" for c in COLS))
        for cell in GRID:
            g = _pick(metrics[cell], group)
            cells = "".join(f"{g.get(c, float('nan')):>10.4f}" for c in COLS)
            print(f"{cell:<20}{g['n_scored']:>3} {cells}")
        delta = "".join(
            f"{_pick(metrics[cell], group).get(c, 0) - base.get(c, 0):>+10.4f}"
            if cell != BASELINE else " " * 10
            for cell in GRID for c in COLS
        )
        print(f"{'  (Δ vs 基线)':<20}{'':>3} {delta}")

    print("\n--- 延迟（全部 20 题，4 格同源）---")
    print(f"{'格':<20}{'召回均值':>10}{'召回中位':>10}{'精排均值':>10}{'精排中位':>10}{'合计中位':>10}")
    for cell in GRID:
        l = lat[cell]
        print(f"{cell:<20}{l['recall_mean_ms']:>10.1f}{l['recall_med_ms']:>10.1f}"
              f"{l['rerank_mean_ms']:>10.1f}{l['rerank_med_ms']:>10.1f}{l['total_med_ms']:>10.1f}")


def dump_evidence(metrics: dict, lat: dict, rows_by_cell: dict) -> None:
    """已给：落盘（指纹可溯源 + 逐题明细，免重跑 GPU）。"""
    per_question = {}
    for cell, rows in rows_by_cell.items():
        per_question[cell] = [
            {
                "id": r["id"], "type": r["type"], "unit": r["unit"], "n_gold": len(r["gold"]),
                # ⚠️ 按 unit 分流（卡5 教训：来源级 gold 拿 chunk_id 比会全是空值）
                "hit_ranks": [i for i, h in enumerate(r["hits"], 1)
                              if hit_key(h, r["unit"]) in r["gold"]],
                "hits": [
                    {"rank": i, "chunk_id": h["chunk_id"], "source": h.get("source"),
                     "rerank_score": round(h.get("rerank_score", 0.0), 4),
                     "in_gold": hit_key(h, r["unit"]) in r["gold"], "preview": h["text"][:80]}
                    for i, h in enumerate(r["hits"], 1)
                ],
            }
            for r in rows
        ]

    payload = {
        "experiment": "w2_embedding_x_reranker_2x2",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "RECALL_K": RECALL_K, "RERANK_TOP_K": RERANK_TOP_K, "SOURCE_CAP": SOURCE_CAP,
            "mode": "dense", "metric_layer": "post_dedupe_pre_quota",
            "grid": {k: {"embed_key": v["embed_key"], "reranker_path": v["reranker_path"]}
                     for k, v in GRID.items()},
            "baseline": BASELINE,
        },
        "fingerprints": {
            "testset": TESTSET_TUNE_JSONL.name, "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
            "gold": GOLD_TUNE_JSON.name, "gold_sha256": sha256_file(GOLD_TUNE_JSON),
        },
        "metrics": metrics,
        "latency": lat,
        "per_question": per_question,
    }
    with open(EVIDENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{EVIDENCE_JSON}")


def main() -> None:
    """已给：装配 → 跑 4 格 → 指标/延迟 → 两道闸 → 打印 → 落盘。"""
    records = load_gold()["records"]
    runners = build_runners()
    report_footprint(runners)

    rows_by_cell, lat_by_cell = build_rows(runners, records)
    metrics = {c: evaluate(rows_by_cell[c]) for c in GRID}
    lat = {c: latency_stats(lat_by_cell[c]) for c in GRID}

    print_table(metrics, lat)

    determinism_check(runners, records, metrics[BASELINE])
    baseline_regression_check(metrics[BASELINE])
    dump_evidence(metrics, lat, rows_by_cell)


if __name__ == "__main__":
    main()