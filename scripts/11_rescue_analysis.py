"""实验 | W1 卡8：救回 / 失手案例分析（零 GPU，读卡7 的落盘证据）。

为什么要有这一卡
    卡7 的读数里，in 组的 recall@5 从 0.3271 涨到 0.36875，Δ = +0.04167。
    in 组 8 题、gold 共 24 块 → 1/24 = 0.04167，所以这个「提升」= 【多命中了 1 个 chunk】。
    同时 recall@3 / nDCG@3 / MRR 三列 Δ 全是 0，cross 与 partial 两组八个指标全 1.0。

    一个 4.17 pt 的 Δ 到底算不算发现，取决于那 1 块是【真答案】还是【页码相邻的邻居】，
    以及它【有没有活到进 prompt 的 3 条】。这不是指标层能回答的 —— 指标只给聚合数，
    要看到「哪一块、排第几、RRF 给了它多少分、最后进没进 top-3」，只能做案例级 diff。
    所以本卡不重跑任何模型：卡7 落盘的逐题明细已经够用。

一处口径声明（别在报告里越界）
    归因标签里最想要的是「这块是不是 BM25 独有的」。但卡7 落盘的 hits 是
    【精排 + 去重后】的列表，不是【召回池】。所以「某块不在 dense 的落盘全量里」
    蕴含两种可能：① dense 根本没召回它（真·BM25 独有）；
    ② dense 召回了，但 dedupe_adjacent 把它当同页重叠冗余删掉了。
    这两者用现有证据区分不了 —— 所以标签只能写到「dense 可用证据里没有」，
    不能写成「BM25 独有」。要真区分，得让卡7 额外落盘召回池，那是另一轮改动。

用法（项目根 rag-kb-next 下，秒级，不需要 GPU）：
    & "<项目 venv>\\Scripts\\python.exe" scripts/11_rescue_analysis.py
    前置：已跑过 scripts/10_hybrid_ab.py，且 data/eval_w1_ab.json 存在
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import GOLD_TUNE_JSON, RERANK_TOP_K, SOURCE_CAP  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.retriever import select_diverse  # noqa: E402

EVIDENCE_JSON = ROOT / "data" / "eval_w1_ab.json"
MODES = ("dense", "hybrid")


# ------------------------------------------------------------
# TODO-1：集合 diff
# ------------------------------------------------------------
def classify(dense_hits: list[dict], hybrid_hits: list[dict], gold: set[str]) -> dict:
    """TODO-1：把一道题的两个 mode 结果分成三类。

    定义（写死，别在实现里含糊）：
        rescued = gold ∩ hybrid - dense   ← hybrid 新增命中的 gold 块（本轮唯一的收益来源）
        lost    = gold ∩ dense - hybrid   ← dense 有、hybrid 弄丢了（代价）
        missed  = gold - dense - hybrid   ← 两组都没捞到（这两组都够不着，是 recall 的天花板）

    返回：
        {"rescued": [(chunk_id, rank), ...],
         "lost":    [(chunk_id, rank), ...],
         "missed":  [chunk_id, ...]}
        rescued/lost 的 rank 是【在该块所属那一组】里的名次：
        救回块给 hybrid 里的名次（它进了哪），失手块给 dense 里的名次（它原本在哪）。

    提示：名次从哪儿来 —— 读 hits[i]["rank"]（卡7 落盘时按顺序写进去的）。
        但读之前加一条断言 `hits[i]["rank"] == i + 1`：
        它把「列表顺序 == 名次」这个前提钉死在代码里。没有这条断言，
        将来谁重排了列表却忘了同步 rank 字段，救回名单会安静地错掉。
    """
    # 前提断言：列表顺序 == 名次。reference 就是 rank 字段，不一致说明落盘被重排过
    for label, hits in (("dense", dense_hits), ("hybrid", hybrid_hits)):
        for i, h in enumerate(hits):
            if h["rank"] != i + 1:
                raise AssertionError(
                    f"{label} 的 rank 字段与列表顺序脱节：#{i + 1} 位置上是 rank={h['rank']}")
            if h["chunk_id"] in gold and not h["in_gold"]:
                raise AssertionError(f"{label} {h['chunk_id']} 的 in_gold 标记与 gold 集合矛盾")

    d_rank = {h["chunk_id"]: h["rank"] for h in dense_hits}
    h_rank = {h["chunk_id"]: h["rank"] for h in hybrid_hits}

    rescued = [(cid, h_rank[cid]) for cid in gold if cid in h_rank and cid not in d_rank]
    lost = [(cid, d_rank[cid]) for cid in gold if cid in d_rank and cid not in h_rank]
    missed = [cid for cid in gold if cid not in d_rank and cid not in h_rank]

    # gold 是 set → 遍历顺序不保证 → 必须排序，否则同一份数据两次跑出的救回名单顺序可能不同
    return {
        "rescued": sorted(rescued, key=lambda t: t[1]),
        "lost": sorted(lost, key=lambda t: t[1]),
        "missed": sorted(missed),
    }


# ------------------------------------------------------------
# TODO-2：零 GPU 重建配额后的 top-k
# ------------------------------------------------------------
def rebuild_top_k(hits: list[dict], k: int, cap: int) -> list[str]:
    """TODO-2：用落盘 hits 重建【配额 + 截断后】的 top-k chunk_id 列表。

    为什么能重建而不重跑：select_diverse 只读 hits[i]["source"] 和【列表顺序】，
    而落盘的 hits 里两个都有（顺序就是精排降序）。所以这条路径不碰 GPU，且
    直接复用已冻结的卡7 证据 —— 比重跑一遍更可信（重跑会引入新变量）。

    要求：
      a) 直接调 select_diverse(hits, k, cap)。⚠️ 别自己重写配额逻辑 ——
         配额只能有一份实现，重写就等于把「两组共用同一后处理」这条前提拆了
      b) 返回 [h["chunk_id"] for h in 结果]
      c) 两条自洽断言（重建可信度的全部依靠）：
         ① len(结果) == min(k, len(hits)) —— 候选池不够 k 条时该给几条就给几条
         ② hits 本身确实按精排降序：scores == sorted(scores, reverse=True)
            ⚠️ 别写成「结果[0] == hits[0]」—— 那是【恒真式】：select_diverse 第一轮循环
               第一个遇到的就是 hits[0]，而它必然被收下（used 为空，0 < cap），
               所以这条断言永远不可能失败 —— 永远不会失败的断言不是检查，是仪式。
               而「列表顺序 == 精排名次」才是重建成立的唯一前提：顺序若被谁重排过，
               配额结果会安静地全错，配额逻辑自己完全察觉不到。
               （已实测：卡7 落盘的 32 组 hits 全部按 rerank_score 降序，这条断言会通过）
         断言失败 = 落盘顺序不再是精排降序，或 select_diverse 的签名/语义变了：
         两种情况都必须立刻停，否则下面的「活没活到 top-3」全是错的
    """
    resout = select_diverse(hits, k, cap)
    if len(resout) != min(k, len(hits)):
        raise AssertionError(f"配额后长度 {len(resout)} != min(k={k}, len(hits)={len(hits)})")
    # ⚠️ 键名是 rerank_score，不是 score —— 落盘时 dump_evidence 只写了 rerank_score
    scores = [h["rerank_score"] for h in hits]
    if scores != sorted(scores, reverse=True):
        raise AssertionError("hits 不是精排降序 —— 重建的 top-k 不可信，停")
    return [r["chunk_id"] for r in resout]


# ------------------------------------------------------------
# TODO-3：归因标签
# ------------------------------------------------------------
def attribute(kind: str, cid: str, rank: int, dense_hits: list[dict],
              own_top_k: list[str]) -> str:
    """TODO-3：给一个 rescued / lost 案例贴一句话标签。

    参数：
        kind        "rescued" 或 "lost"
        cid         涉案的 gold 块
        rank        它在【自己那一组】全量列表里的名次
        dense_hits  dense 组这份落盘全量（用来判断「dense 到底有没有这块」）
        own_top_k   【该案例所属那一组】重建出来的 top-k：
                    rescued 传 hybrid 的，lost 传 dense 的（判断它活没活到 prompt）

    要写出的两层判断：
      第一层 — 这块在 dense 的可用证据里吗？
         在  → "RRF 挪动了名次"（dense 本来就有它，只是位置/是否入榜变了）
         不在 → "BM25 新增来源" —— ⚠️ 但措辞必须写成
                "dense 可用证据里没有"，不能写成 "BM25 独有"：
                不在【精排去重后】全量里，蕴含「dense 没召回」和「被 dedupe_adjacent
                当重叠冗余删了」两种可能，卡7 没落盘召回池，区分不了（见文件头声明）。
                措辞就是结论的边界 —— 过度声明比结论弱更糟
      第二层 — 它活到 top-k 了吗？
         cid in own_top_k → "活到 top-k ✅"
         否则             → f"未活到 top-k（全量 #{rank}）"
        这一层才是救回的实际价值：没进 top-k 的救回，对生成层等于没救

    返回：一句话标签（要能直接进报告），不要返回 bool。
    """
    # 第一层：这块在 dense 的可用证据里吗？（可用证据 = 落盘的精排去重后全量，不是召回池）
    in_dense = any(h["chunk_id"] == cid for h in dense_hits)
    if kind == "rescued":
        origin = ("RRF 挪动了名次（dense 可用证据里本来就有它）" if in_dense
                  else "BM25 新增来源（dense 可用证据里没有）")
    else:
        origin = "hybrid 把 dense 已有的块挤出了全量列表"

    # 第二层：活到 top-k 了吗 —— 这一层才是救回的实际价值
    if kind == "rescued":
        fate = (f"活到 top-{RERANK_TOP_K} ✅" if cid in own_top_k
                else f"未活到 top-{RERANK_TOP_K}（全量 #{rank}）→ 对生成层价值为 0")
    else:
        fate = (f"dense 侧它本来在 top-{RERANK_TOP_K} 内" if cid in own_top_k
                else f"dense 侧它也只到全量 #{rank}")
    return f"{origin}；{fate}"


# ------------------------------------------------------------
# 以下为已给：读证据 / 报告
# ------------------------------------------------------------
def load_evidence() -> tuple[dict, dict, dict]:
    """已给：读卡7 落盘 + gold 真相源，按题号对齐，并校验指纹。"""
    if not EVIDENCE_JSON.exists():
        raise FileNotFoundError(f"{EVIDENCE_JSON} 不存在 —— 先跑 scripts/10_hybrid_ab.py")
    with open(EVIDENCE_JSON, encoding="utf-8") as f:
        ev = json.load(f)

    # 溯源：gold 之后若被改过（页级口径变了），卡7 的指标就不可比，本分析也不成立
    if ev["fingerprints"]["gold_sha256"] != sha256_file(GOLD_TUNE_JSON):
        raise RuntimeError("gold_tune.json 已被改动，卡7 落盘证据失效 —— 重跑 10_hybrid_ab.py")

    records = {r["id"]: r for r in load_gold()["records"] if r["gold"]}
    by_id = {m: {q["id"]: q for q in ev["per_question"][m]} for m in MODES}
    return ev, records, by_id


def main() -> None:
    """已给：逐题 diff → 只打有差异的题 → 汇总。"""
    ev, records, by_id = load_evidence()
    p = ev["params"]
    print(f"证据：{EVIDENCE_JSON.name}（生成于 {ev['generated_at']}）")
    print(f"参数：RECALL_K={p['RECALL_K']} RRF_K={p['RRF_K']} RERANK_TOP_K={p['RERANK_TOP_K']}"
          f" SOURCE_CAP={p['SOURCE_CAP']} | 指标层={p['metric_layer']}")
    print(f"本卡重建层：配额 + 截断到 RERANK_TOP_K={RERANK_TOP_K}（复用 select_diverse）\n")

    n_rescued = n_lost = n_missed = n_gold_total = 0
    n_same_top3 = 0
    n_same_gold_ranks = 0
    n_with_diff = 0

    for qid, rec in records.items():
        gold = rec["gold"]
        d_hits = by_id["dense"][qid]["hits"]
        h_hits = by_id["hybrid"][qid]["hits"]
        d_topk = rebuild_top_k(d_hits, RERANK_TOP_K, SOURCE_CAP)
        h_topk = rebuild_top_k(h_hits, RERANK_TOP_K, SOURCE_CAP)

        n_gold_total += len(gold)
        n_same_top3 += (d_topk == h_topk)
        # 指标是「gold 落在那几个名次」的函数，与填充块无关 —— 所以要单独数这个
        n_same_gold_ranks += ([i for i, c in enumerate(d_topk, 1) if c in gold]
                              == [i for i, c in enumerate(h_topk, 1) if c in gold])

        cls = classify(d_hits, h_hits, gold)
        n_rescued += len(cls["rescued"])
        n_lost += len(cls["lost"])
        n_missed += len(cls["missed"])
        if not (cls["rescued"] or cls["lost"]):
            continue
        n_with_diff += 1

        print(f"[id {qid}] {rec['type']} | {rec['expect']} | gold {len(gold)} 块"
              f" | 全量 {len(d_hits)} / {len(h_hits)} 条 | top{RERANK_TOP_K}"
              f" {d_topk} vs {h_topk}")
        for cid, rank in cls["rescued"]:
            tag = attribute("rescued", cid, rank, d_hits, h_topk)
            h = next(h for h in h_hits if h["chunk_id"] == cid)
            print(f"  ✔ 救回  {cid}  dense: —   hybrid: #{rank}"
                  f"  rrf={h.get('rrf', 0):.5f}  {tag}")
        for cid, rank in cls["lost"]:
            tag = attribute("lost", cid, rank, d_hits, d_topk)
            print(f"  ✘ 失手  {cid}  dense: #{rank}   hybrid: —  {tag}")
        if cls["missed"]:
            print(f"         两组都没捞到：{cls['missed']}")
        print()

    print("=== 汇总 ===")
    print(f"救回 {n_rescued} 块 / 失手 {n_lost} 块 / 两组都没捞到 {n_missed} 块"
          f"（16 题 gold 共 {n_gold_total} 块）")
    print(f"有差异的题（全量层有救回/失手）：{n_with_diff} / {len(records)}")
    print(f"两组 top{RERANK_TOP_K} 序列逐位相同的题数：{n_same_top3} / {len(records)}")
    print(f"两组 top{RERANK_TOP_K} 里【gold 命中位置】相同的题数：{n_same_gold_ranks} / {len(records)}")
    print(f"   ↑ 上面两个数字【必须分开看】：指标只是「gold 落在哪几名」的函数，"
          f"非 gold 的填充块换了多少它完全看不见。\n"
          f"     卡7 表里 @3 全列 Δ=0 的真正依据是后一个数字，不是前一个。")


if __name__ == "__main__":
    main()