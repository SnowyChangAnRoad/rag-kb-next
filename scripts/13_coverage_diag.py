"""实验 | W2 卡0：召回覆盖诊断（零 API；GPU 只用于 dense 编码 8 个 query）。

要回答的问题
    W1 卡8 给出「两路合计都够不着的 gold 块 = 32 块（占 43 块的 74%）」。
    但这个 74% 只是【上界】，它没法区分下面两种完全不同的病因：

        A. 真·覆盖失败     —— 两路召回池里都没有它 → 只有改 embedding / 切分粒度能救
        B. 去重吃掉        —— 召回到了，但 dedupe_adjacent 当同页重叠冗余删掉了 → 改去重规则能救
        C. 精排/截断问题   —— 活到 post-dedupe 全量，但没进 top-3 → 改 RERANK_TOP_K 能救
        D. 其实已交付      —— 进了某个 mode 的 top-3（口径问题，不是失败）

    为什么必须先分清（这是本卡存在的唯一理由）：
        卡4「换 embedding」和卡5「Contextual Retrieval」两卡要花掉 W2 的大头
        （下载 ~3.5GB 模型 + 重建 932 块索引 + 全部 GPU 跑分）。
        如果 74% 里大部分其实是 B/C（改规则就能救），那这两卡的预期收益会被高估，
        做完才发现「A 类只剩十几块」—— 花的钱买的是本来就拿得到的东西。

    证据分两层，必须合并看：
      · 召回池（pre-dedupe）  —— 卡7 没落盘，本脚本现场重跑（8 题，秒级）
      · 落盘证据（post-dedupe）—— data/eval_w1_ab.json，卡7 的原始输出，不重算
    两条口径都用，才能把「召回到没有」和「去重后还在不在」分开数。

用法（项目根 rag-kb-next 下）：
    & "C:\\Users\\xx\\Desktop\\python\\.venv\\Scripts\\python.exe" scripts/13_coverage_diag.py
    前置：data/eval_w1_ab.json 存在（卡7 已跑）；day6_kb 已建
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import bm25_search  # noqa: E402
from rag.config import CHUNKS_JSONL, RECALL_K, RERANK_TOP_K  # noqa: E402
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold  # noqa: E402
from rag.retriever import get_collection, retrieve  # noqa: E402

EVIDENCE_JSON = ROOT / "data" / "eval_w1_ab.json"
MODES = ("dense", "hybrid")

# 块的四个状态，按优先级从好到坏 —— 每块只会落进一类
STATE_TOP3 = "D_top3"          # 进了某个 mode 的 top-3（已交付给 LLM）
STATE_KEPT = "C_kept"          # 在 post-dedupe 全量里，但没进 top-3
STATE_DEDUPED = "B_deduped"    # 在召回池里，但两个 mode 的 post-dedupe 全量都没有
STATE_UNREACHED = "A_unreached"  # 两路召回池都没有（真·覆盖失败）
STATES = (STATE_TOP3, STATE_KEPT, STATE_DEDUPED, STATE_UNREACHED)


def load_evidence() -> dict:
    with open(EVIDENCE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_char_counts() -> dict[str, int]:
    """chunk_id -> char_count。短块分布是「A 类是不是页尾/水印噪声」的直接证据。"""
    out = {}
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                out[c["chunk_id"]] = c["char_count"]
    return out


def evidence_index(evidence: dict) -> dict:
    """(id, mode) -> {"kept": [...post-dedupe 全量], "top3": [...前 3]}。"""
    idx = {}
    for mode in MODES:
        for r in evidence["per_question"][mode]:
            ids = [h["chunk_id"] for h in r["hits"]]
            idx[(r["id"], mode)] = {"kept": ids, "top3": ids[:RERANK_TOP_K]}
    return idx


def diagnose_one(collection, question: str, gold: set[str], ev: dict) -> dict:
    """跑一题，把它的 gold 块分成四类。

    返回 {"states": {chunk_id: state}, "pool": {...}, "n_gold": int, "page_chunks": set}
    """
    # 召回池：现场重跑（pre-dedupe）。dense 走 GPU，BM25 走 CPU。
    dense_pool = [h["chunk_id"] for h in retrieve(collection, question, RECALL_K)]
    bm25_pool = [h["chunk_id"] for h in bm25_search(question, RECALL_K)]
    pool = set(dense_pool) | set(bm25_pool)

    kept = set(ev["kept"])
    top3 = set(ev["top3"])

    states = {}
    for cid in gold:
        if cid in top3:
            states[cid] = STATE_TOP3
        elif cid in kept:
            states[cid] = STATE_KEPT
        elif cid in pool:
            states[cid] = STATE_DEDUPED
        else:
            states[cid] = STATE_UNREACHED

    # 同页兄弟：A 类块的「页」有没有被摸到？只看 dense 池（BM25 池是词面匹配，参考价值低）。
    # 若某块的 doc_id 前缀（页）在 dense 池里出现过其它块，说明检索摸到了这一页，
    # 只是没摸到这一块 —— 那是 chunk 级失败，不是 page 级失败。
    dense_pool_docs = {cid.rsplit("_c", 1)[0] for cid in dense_pool}
    return {"states": states, "dense_pool": dense_pool, "bm25_pool": bm25_pool,
            "n_gold": len(gold), "dense_pool_docs": dense_pool_docs}


def main() -> None:
    evidence = load_evidence()
    ev_idx = evidence_index(evidence)
    records = [r for r in load_gold()["records"] if r["type"] == "in"]  # in 组 = 页级 gold
    char_count = load_char_counts()
    collection = get_collection()

    print(f"卡0 覆盖诊断 | in 组 {len(records)} 题 | RECALL_K={RECALL_K} RERANK_TOP_K={RERANK_TOP_K}")
    print(f"证据：{EVIDENCE_JSON.name}（卡7 落盘，post-dedupe）+ 现场重跑召回池（pre-dedupe）\n")

    # warmup：第一次调用含 CUDA kernel 编译，结果丢弃
    retrieve(collection, records[0]["question"], RECALL_K)

    agg = {s: [] for s in STATES}
    rows = []
    for r in records:
        ev = ev_idx[(r["id"], "dense")]          # dense 是当前服务默认路径
        d = diagnose_one(collection, r["question"], r["gold"], ev)
        for cid, s in d["states"].items():
            agg[s].append(cid)
        rows.append({**d, "id": r["id"], "detail": r["detail"], "question": r["question"]})
        n = {s: sum(1 for v in d["states"].values() if v == s) for s in STATES}
        print(f"id={r['id']:<3} 第{r['detail']['page']}页 gold={d['n_gold']} "
              f"| top3={n[STATE_TOP3]} kept={n[STATE_KEPT]} "
              f"deduped={n[STATE_DEDUPED]} unreached={n[STATE_UNREACHED]}")

    total = sum(len(v) for v in agg.values())
    print(f"\n{'=' * 74}\nin 组 gold 块总数 = {total}")
    for s in STATES:
        cids = agg[s]
        print(f"  {s:<14} {len(cids):>3} 块 ({len(cids) / total * 100:>5.1f}%)")

    # ---- A 类（真·未召回）细看：短块？同页兄弟进池了没有？ ----
    un = agg[STATE_UNREACHED]
    if un:
        cc = sorted(char_count.get(c, 0) for c in un)
        print(f"\nA 类（真·未召回）{len(un)} 块的 char_count："
              f"min={cc[0]} 中位={cc[len(cc) // 2]} max={cc[-1]}")
        short = [c for c in un if char_count.get(c, 0) < 150]
        print(f"  其中 <150 字的短块：{len(short)} 块 "
              f"（W1 已知局限①「页尾 + 广告水印」短块的表现）")
        sibling = 0
        for row in rows:
            for cid, s in row["states"].items():
                if s == STATE_UNREACHED and cid.rsplit("_c", 1)[0] in row["dense_pool_docs"]:
                    sibling += 1
        print(f"  其中「同页有其它块进了 dense 池」的：{sibling} 块 "
              f"→ 这部分是 chunk 级失败（页摸到了，块没摸到）")

    # ---- B 类（去重吃掉）：这些是【改去重规则】就能拿回的，不该指望换 embedding ----
    dd = agg[STATE_DEDUPED]
    if dd:
        print(f"\nB 类（被 dedupe 删）{len(dd)} 块，前 10 个：")
        for c in dd[:10]:
            print(f"    {c:<28} char_count={char_count.get(c, '?')}")

    # ---- 每题一行的「交付层可答性」：top-3 里有没有该页的任何一块 ----
    print(f"\n--- 每题交付层（top-3 是否含该页任一块）---")
    blind = 0
    for row in rows:
        n_top3 = sum(1 for s in row["states"].values() if s == STATE_TOP3)
        flag = "整页落空 ←" if n_top3 == 0 else f"覆盖 {n_top3}/{row['n_gold']}"
        if n_top3 == 0:
            blind += 1
        print(f"  id={row['id']:<3} {flag}")
    print(f"→ {blind}/{len(rows)} 题的 top-3 里【一块 gold 都没有】（真正答不出的题）")

    print(f"\n{'=' * 74}\n结论怎么用：")
    print("  · A 类占比高 → 卡4/卡5（embedding / Contextual）才值得投入，它们改的是【召回】")
    print("  · B 类占比高 → 先改去重规则（零 GPU 成本），别急着换模型")
    print("  · C 类占比高 → 卡4 之外先看 RERANK_TOP_K / 配额（也是零成本）")
    print("  · D 类占比高 → 「74% 够不着」被高估，指标口径比检索质量更值得先修")


if __name__ == "__main__":
    main()