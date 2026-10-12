"""实验 | W2 卡6：MRL 截断对照（qwen3-0.6b：1024 → 512 → 256 维）。

卡型：分析（预测先行）—— 脚本全给，学员的任务是【跑 + 对照预测】，不是填空。

要回答的问题
    卡4 判了 qwen3「无净增益」，那只否掉了它的【精度】价值。它还有第二个价值：MRL。
    把 1024 维截到 512 / 256，召回质量掉多少？换来的存储/延迟收益有多少？
    在 932 块的规模上，这笔交换划不划算？

为什么不改 16_model_ab.py 而是新开一个
    16_model_ab.py 已经跑完并落了卡4 的证据（data/eval_w2_model_ab.json），它现在是
    「已验收的落盘证据生成器」。改它 = 卡4 的那份数字不可复现。所以本脚本自成一份，
    纪律与 16 逐条对齐（下面 4 条），代价是与它长得像 —— 这是刻意付的，不是疏忽。

实验纪律（逐条对齐 16_model_ab.py / 10_hybrid_ab.py）
    1) 单一变量：3 格只有【截断维度】不同，同一份权重、同一套池化与指令、同一条代码路径。
       ⚠️ 本脚本不许出现 if cell 分支；variant 一律由 GRID 逐格传入。
    2) 指标层：post-dedupe pre-quota（精排 + 去重后、配额截断前的全量降序列表）。
    3) 热态：warmup 先行，再【外层题 / 内层格】交替。
    4) 确定性：temperature=0 + 指标不重排 → 重跑基线格应逐位同分。

⚠️ 前提（卡6 的接线改动，已随本卡落地）
    · config.EMBED_MODELS[...]["variants"] 带 {dim, collection}；qwen3 有 raw/512/256 三档；
    · variant 已从 get_collection 一路透传到 encode_query / retrieve / hybrid_retrieve。

用法（项目根 rag-kb-next 下，需 GPU + 关闭沙箱）
    & "<项目 venv>\\Scripts\\python.exe" scripts/18_mrl_ab.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rag.config import (  # noqa: E402
    GOLD_TUNE_JSON, RECALL_K, RERANKER_MODEL_PATH, RERANK_TOP_K, SOURCE_CAP,
    TESTSET_TUNE_JSONL, resolve_variant,
)
from rag.embedding import encode_query, get_embedder  # noqa: E402
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.ingest import build_index, load_chunks  # noqa: E402
from rag.metrics import evaluate  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

# ------------------------------------------------------------
# 对照轴：3 格，唯一变量 = variant（它同时决定"查询编码截到几维"与"查哪个库"）
# ------------------------------------------------------------
EMBED_KEY = "qwen3-0.6b"
GRID = {
    "qwen3__1024": "raw",      # 基线：模型原生维度，与卡4 的 qwen3__base 同一个库
    "qwen3__512": "512",
    "qwen3__256": "256",
}
BASELINE = "qwen3__1024"
CARD4_JSON = ROOT / "data" / "eval_w2_model_ab.json"    # 卡4 落盘，用它的 qwen3__base 当存档闸
EVIDENCE_JSON = ROOT / "data" / "eval_w2_mrl.json"
COLS = ["recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10"]
GROUP_LABEL = {
    "in": "in 题（页级 gold）",
    "cross": "cross 题（来源级 gold）",
    "partial": "partial 题（来源级 gold）",
    "all_with_gold": "全题混合（⚠️ 粒度/题型都不同，仅供参照）",
}


# ------------------------------------------------------------
# 建库：三档都要有 932 块（缺的当场补，已存在的不重建）
# ------------------------------------------------------------
def ensure_collections() -> dict:
    """已给：确保三档库各 932 块，返回 {格: collection}。

    分档策略（不是同一条规则，理由写在这里）：
      · raw 档（qwen3_kb）= 14_embed_check.py 建的、卡4 也在用 —— 本脚本【只读】。
        块数不等于真相源就报错，绝不在这里补写：补写会把卡4 的证据基础改掉。
      · 512 / 256 档是本卡新开的库 —— 块数不对就 upsert 补齐（upsert 幂等，重跑安全）。
      已满 932 的一律跳过，避免白烧一遍 GPU 编码。
    """
    chunks = load_chunks()
    n_expect = len(chunks)
    cols = {}
    for g, variant in GRID.items():
        col = get_collection(model_key=EMBED_KEY, variant=variant)
        n = col.count()
        if variant == "raw":
            if n != n_expect:
                raise AssertionError(
                    f"{g} 的库 {n} 块 != 真相源 {n_expect} 块 —— 它是卡4 的证据基础，"
                    f"本脚本不许改它，请先跑 scripts/14_embed_check.py 重建")
        elif n != n_expect:
            print(f"[建库] {g:<12} variant={variant:<4} count={n} → upsert 补齐 {n_expect} 块…")
            n = build_index(col, chunks)
        cols[g] = col
        print(f"[建库] {g:<12} variant={variant:<4} "
              f"库名={resolve_variant(EMBED_KEY, variant)['collection']:<14} count={n}")
    return cols


def check_dims(cols: dict) -> None:
    """已给：三道断言，把「查询侧截断 == 建库侧截断」变成机器检查而不是人眼。

    ① 查询向量长度 == 该档声明的 dim
    ② 库里的向量长度 == 该档声明的 dim（拿库里第一块向量量一下）
    ③ 反例：拿 1024 维向量去查 512 库 —— 必须【报错】。这道最要紧：
       它证明"维度传错"是响亮失败；真正危险的是"维度相同但截断位置不同"，那没有断言能救，
       只能靠 variant 逐格透传（结构）拦住。
    """
    for g, variant in GRID.items():
        dim = resolve_variant(EMBED_KEY, variant)["dim"]
        qv = encode_query(EMBED_KEY, ["维度自检"], variant)[0]
        assert len(qv) == dim, f"{g}: 查询向量 {len(qv)} 维 != 声明 {dim} 维"

        got = cols[g].get(limit=1, include=["embeddings"])["embeddings"][0]
        assert len(got) == dim, f"{g}: 库内向量 {len(got)} 维 != 声明 {dim} 维"
        print(f"[维度] {g:<12} 查询 {len(qv)} 维 | 库内 {len(got)} 维 | 声明 {dim} 维  ✓")

    qv1024 = encode_query(EMBED_KEY, ["反例"], "raw")[0]
    try:
        cols["qwen3__512"].query(query_embeddings=[qv1024], n_results=1)
    except Exception as e:                      # noqa: BLE001 —— 就是要抓住它
        print(f"[维度] 反例：1024 维查 512 库 → 已拦下（{type(e).__name__}）✓")
        return
    raise AssertionError("1024 维向量查 512 库竟然没报错 —— 保护失效，先查 Chroma 版本行为")


# ------------------------------------------------------------
# 检索段（三格走同一行代码，只有 variant 这个实参不同）
# ------------------------------------------------------------
def retrieve_stage(collection, ranker, question: str, variant: str):
    """已给：召回 → 精排 → 去重，返回 (hits, recall_ms, rerank_ms)。

    ⚠️ variant 必须逐格透传：漏传 → 查询按 1024 维编码 → 查 512 库直接报错
       （响亮，但不是我们要的错）；若哪天维度恰好相同，它就会变成静默失准。
    """
    t0 = time.perf_counter()
    hits = hybrid_retrieve(collection, question, RECALL_K, mode="dense",
                           model_key=EMBED_KEY, variant=variant)
    t1 = time.perf_counter()
    ranked = ranker.rerank(question, hits)
    deduped = dedupe_adjacent(ranked)
    t2 = time.perf_counter()
    return deduped, (t1 - t0) * 1000, (t2 - t1) * 1000


def hit_key(h: dict, unit: str):
    """已给：命中判据按 gold 口径分流。

    ⚠️ 卡5 的教训：cross/partial 是【来源级】gold，拿 chunk_id 去比永远不命中 ——
       指标层没事（evaluate 按 unit 分流），但进度行与证据文件会全是 0，属误导性空值。
    """
    return h["chunk_id"] if unit == "page" else h.get("source")


# ------------------------------------------------------------
# 主循环：warmup + 逐题交替
# ------------------------------------------------------------
def build_rows(runners: dict, records: list[dict]):
    """已给：跑完 3 格，返回 (rows_by_cell, lat_by_cell)。

    硬性要求（与 16_model_ab.py 一一对应）：
      a) warmup：records[0] 三格各跑一次，丢弃 / 不计时 / 不入表
      b) 逐题交替：外层 for 题、内层 for 格
      c) gold 为空的题（out）：hits 照跑（延迟是真实流量），但不进 rows
      d) 延迟口径 = 全部 20 题、三格同源
    """
    rows_by_cell: dict[str, list] = {g: [] for g in GRID}
    lat_by_cell: dict[str, list] = {g: [] for g in GRID}

    for g in GRID:
        retrieve_stage(runners[g][0], runners[g][1], records[0]["question"], GRID[g])

    for r in records:
        for g in GRID:
            hits, t_recall, t_rerank = retrieve_stage(
                runners[g][0], runners[g][1], r["question"], GRID[g])
            lat_by_cell[g].append((t_recall, t_rerank))
            if r["gold"]:
                rows_by_cell[g].append({**r, "hits": hits})   # {**r, ...} 防三格共享同一 dict

            n_hit = sum(1 for h in hits if hit_key(h, r["unit"]) in r["gold"])
            print(f"{g:<12} id={r['id']:<3} 命中 {n_hit}/{len(r['gold'])}  "
                  f"召回 {t_recall:.1f}ms  精排+去重 {t_rerank:.1f}ms")

    return rows_by_cell, lat_by_cell


# ------------------------------------------------------------
# 三道闸
# ------------------------------------------------------------
def archive_check(base_summary: dict) -> None:
    """已给：闸一 —— 基线格（1024）必须 == 卡4 落盘的 qwen3__base，逐字段 != 比较。

    为什么这道闸非有不可：本卡的接线动了注册表形态（dim 从模型记录挪进 variant）与
    三处签名（encode_query / retrieve / hybrid_retrieve）。若 1024 格对不上卡4 的同一个库、
    同一个模型的结果，说明接线引入了偏差 —— 那"512 掉了 X pt"就不是截断造成的，结论作废。
    """
    with open(CARD4_JSON, "r", encoding="utf-8") as f:
        ref = json.load(f)["metrics"]["qwen3__base"]

    diffs: list[str] = []
    for group, new_g in base_summary["by_type"].items():
        old_g = ref["by_type"].get(group)
        if old_g is None:
            diffs.append(f"by_type.{group}: 卡4 里没有这个题型")
            continue
        for col, v_new in new_g.items():
            v_old = old_g.get(col)
            if v_old != v_new:                 # ← != 而不是 isclose
                diffs.append(f"{group}.{col}: 卡4={v_old!r} → 本次={v_new!r}")

    if diffs:
        print("存档闸未通过（接线引入了偏差）：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"1024 格 != 卡4 qwen3__base，共 {len(diffs)} 个字段")
    print("存档闸通过：1024 格 == 卡4 的 qwen3__base（接线零偏差）")


def determinism_check(runners: dict, records: list[dict], before: dict) -> None:
    """已给：闸二 —— 重跑基线格，断言指标逐位相同（用 !=，不用 isclose）。"""
    key = BASELINE
    rows = [{**r, "hits": retrieve_stage(runners[key][0], runners[key][1],
                                        r["question"], GRID[key])[0]}
            for r in records if r["gold"]]
    after = evaluate(rows)

    diffs: list[str] = []
    for group, new_g in after["by_type"].items():
        old_g = before["by_type"].get(group)
        if old_g is None:
            diffs.append(f"by_type.{group}: 重跑后多出/少了这个题型")
            continue
        for col, v_new in new_g.items():
            v_old = old_g.get(col)
            if v_old != v_new:
                diffs.append(f"{group}.{col}: {v_old!r} → {v_new!r}")

    if diffs:
        print("确定性自检未通过：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"基线格重跑后指标漂移，共 {len(diffs)} 个字段")
    print("确定性自检通过：基线格重跑后指标【逐位相同】")


# ------------------------------------------------------------
# 以下为已给：存储账 / 延迟统计 / 打印 / 落盘
# ------------------------------------------------------------
def print_storage(cols: dict) -> None:
    """已给：理论向量占用 + 实际落盘目录大小。

    ⚠️ 口径说明（不然这份数字会被误读）：理论值 = count × dim × 4B（cosine 空间下
       Chroma 存 float32）。它在 932 块的规模上是 MB 级 —— MRL 省存储的真正场合是
       百万级语料（那里 1024→256 省的是 GB 级）。所以本表回答的是"能不能省"，
       不是"现在省了多少"。
    """
    print("\n--- 存储（向量理论占用 = count × dim × 4B）---")
    for g, variant in GRID.items():
        dim = resolve_variant(EMBED_KEY, variant)["dim"]
        n = cols[g].count()
        print(f"{g:<12} dim={dim:>5} count={n:>4} "
              f"向量={n * dim * 4 / 1e6:>7.2f}MB "
              f"（相对 1024 维 {n * dim * 4 / (n * 1024 * 4):>5.1%}）")

    db = ROOT / "chroma_db"
    total = sum(p.stat().st_size for p in db.rglob("*") if p.is_file())
    print(f"chroma_db 全目录实际占用 = {total / 1e6:.1f}MB（含全部库与 HNSW 图，非单库值）")


def latency_stats(pairs: list[tuple[float, float]]) -> dict:
    """已给：两段耗时的均值与中位数（中位数对单次 CUDA 抖动免疫）。"""
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
    print(f"对照轴：MRL 截断维度 | 基线格 = {BASELINE}")

    for group in ("in", "cross", "partial", "all_with_gold"):
        base = _pick(metrics[BASELINE], group)
        if not base:
            continue
        print(f"\n--- {GROUP_LABEL[group]} ---")
        print(f"{'格':<14}{'n':>3} " + "".join(f"{c:>10}" for c in COLS))
        for cell in GRID:
            g = _pick(metrics[cell], group)
            cells = "".join(f"{g.get(c, float('nan')):>10.4f}" for c in COLS)
            print(f"{cell:<14}{g['n_scored']:>3} {cells}")
        delta = "".join(
            f"{_pick(metrics[cell], group).get(c, 0) - base.get(c, 0):>+10.4f}"
            if cell != BASELINE else " " * 10
            for cell in GRID for c in COLS
        )
        print(f"{'  (Δ vs 基线)':<14}{'':>3} {delta}")

    print("\n--- 延迟（全部 20 题，3 格同源）---")
    print(f"{'格':<14}{'召回均值':>10}{'召回中位':>10}{'精排均值':>10}{'精排中位':>10}{'合计中位':>10}")
    for cell in GRID:
        l = lat[cell]
        print(f"{cell:<14}{l['recall_mean_ms']:>10.1f}{l['recall_med_ms']:>10.1f}"
              f"{l['rerank_mean_ms']:>10.1f}{l['rerank_med_ms']:>10.1f}{l['total_med_ms']:>10.1f}")


def dump_evidence(metrics: dict, lat: dict, rows_by_cell: dict, storage: dict) -> None:
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
        "experiment": "w2_mrl_truncation_1024_512_256",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "RECALL_K": RECALL_K, "RERANK_TOP_K": RERANK_TOP_K, "SOURCE_CAP": SOURCE_CAP,
            "mode": "dense", "metric_layer": "post_dedupe_pre_quota",
            "embed_key": EMBED_KEY, "grid": GRID, "baseline": BASELINE,
        },
        "fingerprints": {
            "testset": TESTSET_TUNE_JSONL.name, "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
            "gold": GOLD_TUNE_JSON.name, "gold_sha256": sha256_file(GOLD_TUNE_JSON),
        },
        "metrics": metrics,
        "latency": lat,
        "storage": storage,
        "per_question": per_question,
    }
    with open(EVIDENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{EVIDENCE_JSON}")


def main() -> None:
    """已给：建库 → 维度自检 → 装配 → 跑 3 格 → 三道闸 → 打印 → 落盘。"""
    records = load_gold()["records"]
    cols = ensure_collections()
    check_dims(cols)

    ranker = BGEReranker(RERANKER_MODEL_PATH)          # 三格共用同一个精排器（不是变量）
    runners = {g: (cols[g], ranker) for g in GRID}

    rows_by_cell, lat_by_cell = build_rows(runners, records)
    metrics = {c: evaluate(rows_by_cell[c]) for c in GRID}
    lat = {c: latency_stats(lat_by_cell[c]) for c in GRID}

    print_table(metrics, lat)
    print_storage(cols)

    determinism_check(runners, records, metrics[BASELINE])
    archive_check(metrics[BASELINE])

    storage = {
        c: {"dim": resolve_variant(EMBED_KEY, GRID[c])["dim"],
            "count": cols[c].count(),
            "vector_mb": cols[c].count() * resolve_variant(EMBED_KEY, GRID[c])["dim"] * 4 / 1e6}
        for c in GRID
    }
    dump_evidence(metrics, lat, rows_by_cell, storage)
    print(f"显存（三档共用同一份权重）：allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")


if __name__ == "__main__":
    main()