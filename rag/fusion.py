"""检索层 | W1 卡4：RRF 融合（dense ⊕ BM25 → 单一排序）。

为什么要它：dense 和 BM25 的失败模式不相关 —— dense 强在语义改写，
BM25 强在字面 token 精确相等。但两路的分数【不同量纲、不同方向】：
    dense  distance  越小越好，0~2，实测集中在 0.2~0.6
    BM25   score     越大越好，无上界，且随 query 词数增长（卡3 第五条实测）
直接加权相加就得先做 per-query 归一化，等于用一个新参数去还旧债。
RRF 把分数整个扔掉，只用名次：

    RRF(d) = Σ_i  1 / (RRF_K + rank_i(d))        rank 从 1 开始

k=60 时 1/61=0.01639、1/63=0.01587 —— 名次差异只有 1.6%，所以
【谁被两路同时提名谁赢】，而不是「某一路排名极高」的块赢：
    共识块(dense#4 + BM25#1) = 0.0320   >   单路块(仅 dense#2) = 0.0161
推论：那条「dense 漏了、BM25 召回了」的救回块只拿【单路票 0.0161】，
在融合层排不到前面 —— 它必须靠后面的 BGEReranker 捞进 top-3。
⇒ RRF 在精排【之前】，它的职责不是「选出最终答案」，而是「把候选池换成更好的原料」。

本文件三条硬约束：
  1) 只吃 rank，不吃分数。任何「用 distance/score 加权」的写法都退回了归一化难题
  2) 与 retriever.retrieve() 同形同参：返回【长度 k】的 hits。rerank / 去重 / 配额
     仍在 pipeline._retrieve 里 —— 卡9 才能只改一个 config 开关，pipeline 一行都不动
  3) 排序必须【确定性】。同分时不能依赖 dict 插入顺序，否则每跑一次 A/B，
     top-5 都会轻微漂移，指标差异就说不清是实验变量还是随机
     （Day10 把 temperature 从 0.7 改成 0，就是同一个教训）

用法（项目根 rag-kb-next 下，包内代码一律 python -m）：
    python -m rag.fusion        # 自检：同一 query 的 dense / BM25 / 融合 三路对照
"""
from .bm25 import bm25_search
from .config import DEFAULT_EMBED, DEFAULT_VARIANT, RECALL_K, RRF_K
from .retriever import get_collection, retrieve


def hybrid_retrieve(collection, question: str, k: int = RECALL_K,
                    mode: str = "hybrid", model_key: str = DEFAULT_EMBED,
                    variant: str = DEFAULT_VARIANT) -> list[dict]:
    """TODO：dense top-N ⊕ BM25 top-N → RRF → top-k。返回与 retrieve() 同构的 hits。

    参数：
        mode="dense"   只走向量一路（卡7 对照表的【基线组】）
        mode="hybrid"  两路融合（实验组）
        为什么要有 mode，而不是让卡7 脚本自己调两个函数：
            基线组与实验组必须共用【除召回段以外的全部代码路径】。
            两条独立调用链一旦分叉，将来改一处忘另一处，对照表会静默失真 ——
            pipeline._retrieve 的 docstring 反复警告过这件事（检索口径只能有一份）。
        N（每路候选深度）故意用 RECALL_K，不另设常量：
            基线组给的也是 RECALL_K 条，两边同源，才保证卡7 的 A/B
            唯一的实验变量是「召回段有没有 BM25」。

    ------------------------------------------------------------------
    TODO-1：两路召回，各取 RECALL_K 条
    ------------------------------------------------------------------
        dense_hits = ???          # 提示：retrieve() 已经写好，注意它要吃 collection
        bm25_hits  = ???          # 提示：卡3 的 bm25_search(query, k)，不吃 collection

        注意：bm25_search 可能返回【少于】N 条（它过滤了 score <= 0）。
        这不是 bug，是卡3 的刻意决策 —— 所以下面的循环不能假设
        「两路长度一定相等」，更不能按下标配对（两路的下标毫无关系，
        唯一的共同键是 chunk_id）。

    ------------------------------------------------------------------
    TODO-2：RRF 累加（本卡的核心，只有 4 行左右）
    ------------------------------------------------------------------
        建议的累加容器：chunk_id -> {"hit": <该块的 hit 字典>, "rrf": float}

        要写对的 4 件事：
          a) rank 从 1 开始：enumerate(hits, start=1)。从 0 开始会多给一个
             1/(60+0)=0.01667 的"白送分"，且第 1 名和第 0 名撞在一起
          b) 分值是 1/(RRF_K + rank)，【不带任何权重】。想给 BM25 加权重
             （比如 1.5×）就已经不是 RRF 了，那是另一套要调参的方法
          c) 同一条块被两路都命中 → 分值【相加】。这是增益的全部来源，
             忘了相加就得到「两路各排各的、谁在前谁赢」，比单路还差
          d) 两路都给同一条块时，留哪一份 hit？
             dense 的 hit = {"chunk_id","distance","text", **chroma 元数据}
             BM25  的 hit = {"chunk_id","score","text", **扁平字段}（卡3 已改为整块平铺）
             提示三个下游的读法，据此决定覆盖方向，别随手写：
               - ranker.rerank()    只读 h["text"]
               - dedupe_adjacent()  读 h["chunk_id"]
               - select_diverse()   读 h["source"]
               - trace.py           会格式化 h["distance"]:.4f ← 卡9 的雷
             还要决定：要不要在 hit 上保留两路各自的名次
             （如 dense_rank / bm25_rank）？卡8 要做「救回案例」分析，
             需要知道某块「在 BM25 排第几、在 dense 排第几或压根没进池」。

        每条 hit 最终必须带一个 rrf 字段（融合分），供自检与卡7 打印。

    ------------------------------------------------------------------
    TODO-3：确定性排序 + 截断到 k
    ------------------------------------------------------------------
        按 rrf 降序取前 k 条。踩点：
          - sorted(..., key=..., reverse=True) 是稳定排序，但前提是【输入顺序稳定】。
            而 accum 是 dict —— dict 保持插入顺序，而插入顺序 = 先遍历哪一路。
            两路相对顺序反过来，同分块的名次就反过来 → 不可复现
          - 所以必须给一个显式的 tie-break。可选做法：
              ① key 里加一个确定性的第二排序键（如 chunk_id 字典序）
              ② 干脆把 key 写成完整元组，让同分无法"并列"
            挑一种，并写清楚为什么它能保证跨运行一致。
          - 截断用 [:k]，与 retrieve() 口径一致（它也是返回恰好 k 条）。

    返回：长度 ≤ k 的 hits 列表，按 rrf 降序。
         k 条里可能【不足 K】—— 若两路并集就不到 k 条（小库/极端 query），
         不要补位、不要重复，返回实际条数即可。
    """
    # 快速失败：mode 写错时立刻报错，而不是静默走到某个分支
    if mode not in ("dense", "hybrid"):
        raise ValueError(
            f"fusion.hybrid_retrieve: mode 只支持 'dense' / 'hybrid'，收到 {mode!r}"
        )

    # 基线分支（已给）：只走向量一路，与 retrieve() 的回调口径完全一致
    if mode == "dense":
        return retrieve(collection, question, k, model_key, variant)

    # ------------------ TODO-1：两路召回 ------------------
    dense_hits = retrieve(collection, question, RECALL_K, model_key, variant)
    bm25_hits = bm25_search(question, RECALL_K)

    # ------------------ TODO-2：RRF 累加 ------------------
    # 容器：chunk_id -> {"hit": <该块的 hit>, "rrf": float}
    # 为什么不用两个平行 dict：chunk_id 是两路唯一的共同键，用它做 key 天然完成
    # 「同一块合并成一条」；两路的下标毫无关系，绝不能按下标配对。
    accum: dict[str, dict] = {}
    for i, h in enumerate(dense_hits, 1):          # start=1：rank 从 1 开始
        accum[h["chunk_id"]] = {"hit": h, "rrf": 1 / (RRF_K + i)}
    for i, h in enumerate(bm25_hits, 1):
        if h["chunk_id"] in accum:                 # 显式成员判断，不靠 value 的真值
            accum[h["chunk_id"]]["rrf"] += 1 / (RRF_K + i)   # ← 共识增益：两路都命中就相加
        else:
            accum[h["chunk_id"]] = {"hit": h, "rrf": 1 / (RRF_K + i)}
    # 两路都命中时保留【dense 那份 hit】是刻意的：dense hit 带 distance，
    # 且其 meta（ingest.clean_meta 写入）已含 doc_id/source/page/url/title/char_count，
    # 是 BM25 hit 的超集；反过来保留 BM25 那份会丢掉 distance（卡9 的 trace 要读它）。
    # 代价：BM25 的 score 在这条 hit 上没了 —— 卡8 若要分析「BM25 排第几」需另外记录。

    # ------------------ TODO-3：确定性排序 + 截断到 k ------------------
    # key 写成元组 (-rrf, chunk_id)：主序 rrf 降序（用负号翻转），
    # 次序 chunk_id 升序。为什么不靠 sorted 的稳定性：accum 的遍历顺序取决于
    # 「先遍历 dense 再遍历 bm25」这一实现细节，一旦两路顺序被换掉，
    # 同分块的名次会静默翻转，A/B 指标就说不清是实验变量还是这个副作用。
    ranked = sorted(accum.values(), key=lambda it: (-it["rrf"], it["hit"]["chunk_id"]))

    # 把融合分写回 hit —— 自检、卡7 的对照表、卡8 的案例分析都要读它。
    # hit 是本次调用内新建的 dict（retrieve/bm25_search 每次返回新对象），写回不会污染调用方。
    for it in ranked:
        it["hit"]["rrf"] = it["rrf"]
    return [it["hit"] for it in ranked[:k]]



if __name__ == "__main__":
    # 已给：自检入口。要看得到三件事：
    #   ① 两路到底重不重叠（交集大小 = RRF 有没有发挥空间的先决条件）
    #   ② 融合后的 top-5 各自来自哪几路（d#1+b#3 这种）
    #   ③ 只在 BM25 里出现的块有没有被融合排上去（大概率没有 —— 这是正常现象，
    #      它们只有单路票 0.0161，要等精排来捞）
    collection = get_collection()
    for q in ("RRF 融合", "chunkText 的分块策略", "怎么把长文本切成小段"):
        print(f"\n=== query: {q} ===")
        d_ids = [h["chunk_id"] for h in retrieve(collection, q, RECALL_K)]
        b_ids = [h["chunk_id"] for h in bm25_search(q, RECALL_K)]
        inter, union = set(d_ids) & set(b_ids), set(d_ids) | set(b_ids)
        print(f"dense {len(d_ids)} 条 | BM25 {len(b_ids)} 条 | "
              f"交集 {len(inter)} | 并集 {len(union)} | "
              f"只在 dense {len(set(d_ids) - inter)} | 只在 BM25 {len(set(b_ids) - inter)}")

        for h in hybrid_retrieve(collection, q, RECALL_K)[:5]:
            cid, tags = h["chunk_id"], []
            if cid in d_ids:
                tags.append(f"d#{d_ids.index(cid) + 1}")
            if cid in b_ids:
                tags.append(f"b#{b_ids.index(cid) + 1}")
            print(f"  rrf={h['rrf']:.5f}  {'+'.join(tags) or '??':<8} {cid}")

    # 顺手验一下 mode 分支：两者应只在召回段不同
    print("\n=== mode='dense' 应与 retrieve() 完全一致 ===")
    q = "RRF 融合"
    a = [h["chunk_id"] for h in hybrid_retrieve(collection, q, RECALL_K, mode="dense")]
    b = [h["chunk_id"] for h in retrieve(collection, q, RECALL_K)]
    print("一致" if a == b else f"不一致！\n  {a}\n  {b}")