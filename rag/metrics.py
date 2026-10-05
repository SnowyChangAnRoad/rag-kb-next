"""检索层评估 | W1 卡5：recall@k / MRR / nDCG@k（页级 gold 口径）。

为什么要它（W1 计划 §五 第一行）：
    现有 eval.py 只有【生成层】指标（答案正确率 / 引用正确率），它们被生成噪声污染 ——
    20 题里只要有 2 题生成抖动，结论就可能反转（Day9 已实测：11 题里 8 题两组资料逐字相同，
    指标却仍不同）。检索层指标【不吃 LLM】：给定 hits 与 gold，同一份数据重跑永远同分 ——
    和 eval.retrieval_main 是同一个思路，只是那个用的是「期望来源命中」这个粗代理，
    这里换成三个有名字、可对外引用的标准指标。

三个指标各回答一个问题，缺一个就会漏掉一类失败：
    recall@k    gold 里有多少条落进了 top-k   —— 覆盖度，【完全不看顺序】
    MRR         第一个相关块排得多靠前         —— 【只看第一个】，对后续 gold 全盲
    nDCG@k      相关块排得够不够靠前           —— 唯一带位置折扣的

⚠️ 口径绑定（W1 约定，卡 5/6 期间不许改）：
    gold 颗粒度 = 页级。in 题取 expect 的「第 N 页」→ 该页【全部】chunk_id 都算 gold；
    cross 题取文件集合 → 该文件【全部】chunk 算 gold。
    ⇒ 页级必然【高估 recall】（一页 3~5 块，命中任一块即算命中），
      所以 recall@5 只能【同口径横比】（dense-only vs hybrid），不能当对外绝对值。
    ⇒ cross 题的 gold = 整个文件（可能几百块），recall@5 天然只有 0.00x，
      和 in 题的 0.2~0.6【不可加】→ 汇总必须【分题型】做 macro，混合值只作参考。

本文件只做【纯函数】：吃「已排好序的 hits」+「gold 集合」，返回数字。
    不做 IO、不读 chroma、不展开 expect（那是卡6 的活）、不重排 hits ——
    hits 的顺序由 fusion.hybrid_retrieve 的 (-rrf, chunk_id) tie-break 负责；
    在评估层再排一次 = 引入第二个排序口径，A/B 就说不清是谁的效果了。
    纯函数的直接好处：自检能用【合成数据手算】验算公式，公式没锁死之前不碰真实语料，
    否则「公式写错」和「gold 口径写错」会同时暴露，定位成本翻倍。

输入契约（卡6/卡7 会按这个拼数据）：
    hits = list[dict]，每项至少有 "chunk_id"；按名次从好到坏【已排好】
    gold = set[str]，该题全部 gold chunk_id
    rows = list[dict]，每题一行：{"id", "type", "gold": set[str], "hits": list[dict]}

用法（项目根 rag-kb-next 下，包内代码一律 python -m）：
    python -m rag.metrics      # 自检：合成用例 + 手算期望值断言 + 指标手感对照表
"""

import math

# 评估深度：同一份 hits 算三次（零成本、零额外检索），便于看「加深 top-k 还能不能捞回来」。
# 为什么不用 config：它是评估模块自己的口径（和 eval.py 里的 GROUPS 同类），
# 不属于「检索行为」参数 —— config 里放的是运行时会被线上路径读到的值。
EVAL_KS = (3, 5, 10)

# 两种 gold 粒度（卡6 落进 gold 记录的 "unit" 字段）：
#   "page"    in 题：gold = 该页全部 chunk_id，分母 = 页内块数（3~5）→ 分辨率 0.2/0.4/…/1.0
#   "source"  cross/partial 题：gold = 文件名列表，分母 = 文件数（cross=2）→ 分辨率 0/0.5/1
# 卡6 的修订理由写在 rag/gold.py 文件头：计划原文「该文件全部 chunk」会让 cross 的
# 分母变成 401 块，recall@5 天花板 0.012，A/B 的差异全被淹掉。
PAGE_UNIT = "page"
SOURCE_UNIT = "source"


def as_source_hits(hits: list[dict]) -> list[dict]:
    """把 chunk 粒度 hits 投影成【来源粒度】：同一文件只保留【第一次出现】的位置。

    为什么要「只留第一次」而不是把每块都投影一遍：
        后者会让同一文件在名次序列里出现多次 —— recall 用集合交集不受影响（自动去重），
        但 DCG 会把第 1 名和第 2 名的同一个文件算两次折扣，nDCG 被虚高。
        去重后每个文件只有一个名次，语义正是「该文件最好排到第几」。
    投影后的 chunk_id 位置放【文件名】，于是三个纯函数【一行都不用改】就能复用到来源粒度。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        s = h["source"]
        if s not in seen:
            seen.add(s)
            out.append({"chunk_id": s})
    return out


def _project(hits: list[dict], unit: str) -> list[dict]:
    """按该题的 gold 粒度把 hits 对齐到同一粒度：来源级题先投影，页级题原样返回。"""
    return as_source_hits(hits) if unit == SOURCE_UNIT else hits


def first_relevant_rank(hits: list[dict], gold: set[str]) -> int | None:
    """TODO-1：第一个 ∈ gold 的块的名次（从 1 开始）；一个都没命中 → None。

    为什么返回 None 而不是 0：
        0 是个「看起来很合法」的名次，下游顺手写 1/0 会 ZeroDivisionError；
        None 逼调用方显式处理「没命中」这一支（reciprocal_rank 就是把它记 0 分）。
        这是「用类型表达状态」，比注释管用。
    """
    for i, h in enumerate(hits, 1):
        if h["chunk_id"] in gold:
            return i
    return None


def recall_at_k(hits: list[dict], gold: set[str], k: int = 5) -> float:
    """TODO-2：recall@k = |gold ∩ top-k| / |gold|。

    要点（每一条都对应一个会静默出错的写法）：
      - 只看【前 k 条】：hits[:k]。忘了切片就变成「recall@∞」，A/B 里两组都接近 1，
        指标直接失去分辨率，而且不会报错
      - 分母是 |gold|（该题 gold 总数），【不是 k、也不是 len(hits)】。
        用 k 会把「一页只有 3 块 gold」的题算成 3/5，数据全是错的
      - 交集用【集合】算：len(set(h["chunk_id"] for h in hits[:k]) & gold)
        （hits 是 dict，不可 hash；也别 bool 化判断结果）
      - 一个都没命中 → 0.0（不是 None，recall 有明确的零值）
      - 空 gold → 早返回 0.0，避免 ZeroDivisionError
        （正常流程下 evaluate 已把无 gold 题剔除，这里是函数自己的边界安全）
    返回 float，范围 [0, 1]。1.0 = 该题 gold 全在 top-k 里。
    """
    if not gold:                      # 空 gold 早返回：否则 len(gold)=0 → ZeroDivisionError
        return 0.0
    hit = {h["chunk_id"] for h in hits[:k]} & gold   # &取交集
    return len(hit) / len(gold)


def reciprocal_rank(hits: list[dict], gold: set[str]) -> float:
    """TODO-3：RR = 1 / 第一个相关块的名次（单题）。MRR 是它在题集上的平均。

    要点：
      - 只关心【第一个】：第 2 个 gold 排得多靠前都不影响 RR。
        这是 MRR 的定义，也是它最大的局限（面试常问「MRR 什么时候会骗你」：
        答案分散在多块时，检索很成功但后 9 条全被忽略）
      - 一个都没命中 → 0.0。⚠️【必须记 0，不能把这题跳过】——
        跳过会让「检索越差 → 参与平均的题越少 → 平均分越高」，指标方向反转。
        这是汇总层最容易犯的错，evaluate 里的 mrr 分母用的是「有 gold 的题数」
      - 单题上限 1.0（第 1 名就命中）；分母【不含 k】，MRR 没有 @k 的说法
    """
    rank = first_relevant_rank(hits, gold)      # 只扫一次：你的写法调了两次，白扫一遍
    return 1 / rank if rank else 0.0


def dcg_at_k(hits: list[dict], gold: set[str], k: int = 5) -> float:
    """TODO-4：DCG@k = Σ_{i=1..k} (2^rel_i − 1) / log2(i+1)。

    二值口径（本卡已选定）：rel_i = 1 当 hits[i-1] ∈ gold，否则 0。
    于是 gain = 2^1 − 1 = 1 恒成立，公式退化成「命中的位各加一个折扣」：
        DCG@k = Σ_{命中位 i，i ≤ k} 1 / log2(i+1)
    """
    resout = 0.0
    for i, h in enumerate(hits[:k], 1):
        if h["chunk_id"] in gold:
            resout += 1 / math.log2(i+1)
    return resout


def idcg_at_k(n_gold: int, k: int = 5) -> float:
    """TODO-5：IDCG@k = 【理想排序】下的 DCG@k，即 nDCG 的分母。

    理想排序 = 所有 gold 排在最前，因此前 min(n_gold, k) 位都是相关位：
        IDCG@k = Σ_{i=1..min(n_gold, k)} 1 / log2(i+1)

    边界：
      - n_gold = 0 → 0.0（ndcg_at_k 拿它判「这题没有 gold，跳过」）
      - n_gold < k：**不能**按 k 位累加 —— 多算的位在理想排序里也是 0 增益，
        多算会把 nDCG【压低】（把上限抬高到实际达不到的位置）。这是本卡最容易写错的一处。
      - n_gold > k：只加到 k 位（理想排序里第 k 位之后根本不进 top-k）
    """
    resout = 0.0
    for i in range(1, min(n_gold, k) + 1):
        resout += 1 / math.log2(i+1)
    return resout


def ndcg_at_k(hits: list[dict], gold: set[str], k: int = 5) -> float:
    """已给：nDCG@k = DCG@k / IDCG@k。

    唯一的保护是分母判 0：无 gold 的题（out 题）IDCG 也是 0，
    这里返回 0.0 而不是崩溃 —— 但真正的「跳过无 gold 题」发生在 evaluate 汇总层，
    千万别把 0.0 也平均进去（那等于给拒答题打 0 分，见 reciprocal_rank 的说明）。
    """
    ideal = idcg_at_k(len(gold), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(hits, gold, k) / ideal


# ------------------------------------------------------------
# 汇总（已给）：从「一题一行」的明细算出「一组一个数」
# ------------------------------------------------------------
# 为什么必须分题型 macro（W1 约定，见文件头）：
#   in 题 gold = 一页 3~5 块 → recall@5 量级 0.2~0.6
#   cross 题 gold = 整个文件（几百块）→ recall@5 量级 0.00x
#   两者【不同量纲】，mix 在一起平均，cross 的极端小值会把 in 的变化彻底淹没
#   （pooled 更糟：Σ命中/Σgold 里 cross 的几百块分母会直接把总分母撑大，
#     hybrid 救回 1~2 块对总分的影响趋近 0，等于测不出 W1 的增益）。
#   所以：by_type 是主结果，all 只是「有 gold 的题合起来看」的参考值。
def evaluate(rows: list[dict], ks: tuple[int, ...] = EVAL_KS) -> dict:
    """rows: [{"id", "type", "unit", "gold": set[str], "hits": list[dict]}, ...]

    "unit" 是该题的 gold 粒度（卡6 写入）："page" = gold 是 chunk_id 集合；
    "source" = gold 是文件名集合（cross/partial 题）。缺省按 "page" 走 ——
    卡5 的合成用例没带这个字段，保持向后可用。

    返回：
        {
          "by_type": {"in": {...}, "cross": {...}, "partial": {...}},
          "all_with_gold": {...},          # 仅参考，不要当 headline
          "n_no_gold": int,                # 被跳过的题数（out 题）
        }
    每组内部：
        n       该组题数
        n_scored  真正参与打分的题数（有 gold 的）
        mrr     Σ RR / n_scored        ← 未命中记 0 分，不跳题
        recall@k / ndcg@k   同样除以 n_scored（macro）

    ⚠️ 粒度不同的题【只能在同一 by_type 组内比】（in 组是页级、cross 组是来源级）：
       两组的数值不可加、也不该放同一张 headline 里比大小 —— 这正是「分题型」的另一个理由。
    """
    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        scored = [r for r in rs if r["gold"]]
        m = len(scored)
        out = {"n": n, "n_scored": m}
        if not m:
            return out
        # 先按该题粒度对齐 hits，三个指标读的都是对齐后的序列
        proj = [(_project(r["hits"], r.get("unit", PAGE_UNIT)), r["gold"]) for r in scored]
        out["mrr"] = sum(reciprocal_rank(h, g) for h, g in proj) / m
        for k in ks:
            out[f"recall@{k}"] = sum(recall_at_k(h, g, k) for h, g in proj) / m
            out[f"ndcg@{k}"] = sum(ndcg_at_k(h, g, k) for h, g in proj) / m
        return out

    result = {
        "by_type": {t: agg([r for r in rows if r["type"] == t])
                    for t in ("in", "cross", "partial")
                    if any(r["type"] == t for r in rows)},
        "all_with_gold": agg([r for r in rows if r["gold"]]),
        "n_no_gold": sum(1 for r in rows if not r["gold"]),
    }
    return result


# ------------------------------------------------------------
# 自检（已给）：合成数据 + 手算期望值
# ------------------------------------------------------------
# 为什么先跑合成用例再上真实语料：
#   公式的正确性必须先在【能心算的规模】上锁死。真实库 932 块、页级 gold 3~5 条，
#   算出来的数你无法一眼判断对错 —— 一旦结果可疑，你分不清是公式错、gold 错还是语料问题。
def _self_check() -> None:
    gold = {"a", "b", "c"}                                   # 3 条 gold
    # 命中位在第 2、4 名；第 1/3/5 名是干扰块
    hits = [{"chunk_id": cid} for cid in ("x", "a", "y", "b", "z")]

    assert first_relevant_rank(hits, gold) == 2, "第一个相关块应在第 2 名"
    assert math.isclose(reciprocal_rank(hits, gold), 0.5), "RR = 1/2"

    # recall：top-3 只捞到 a（1/3）；top-5 捞到 a、b（2/3）
    assert math.isclose(recall_at_k(hits, gold, 3), 1 / 3)
    assert math.isclose(recall_at_k(hits, gold, 5), 2 / 3)

    # DCG@5 = 1/log2(3) + 1/log2(5)（第 2 位与第 4 位）
    dcg5 = 1 / math.log2(3) + 1 / math.log2(5)
    assert math.isclose(dcg_at_k(hits, gold, 5), dcg5)
    # IDCG@5：|gold|=3 < k=5 → 只加到第 3 位 = 1 + 1/log2(3) + 1/log2(4)
    assert math.isclose(idcg_at_k(3, 5), 1 + 1 / math.log2(3) + 1 / math.log2(4))
    # |gold|=7 > k=5 → 加满 5 位
    assert math.isclose(idcg_at_k(7, 5),
                        1 + 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5) + 1 / math.log2(6))
    assert ndcg_at_k(hits, gold, 5) == dcg5 / (1 + 1 / math.log2(3) + 1 / math.log2(4))

    # 边界：全不命中
    miss = [{"chunk_id": cid} for cid in ("x", "y", "z")]
    assert first_relevant_rank(miss, gold) is None
    assert reciprocal_rank(miss, gold) == 0.0
    assert recall_at_k(miss, gold, 5) == 0.0
    assert ndcg_at_k(miss, gold, 5) == 0.0
    # 边界：无 gold（out 题）—— 三个函数都不能崩
    assert recall_at_k(hits, set(), 5) == 0.0
    assert ndcg_at_k(hits, set(), 5) == 0.0
    assert idcg_at_k(0, 5) == 0.0

    print("公式断言全部通过（含 |gold| < k / |gold| > k / 全不命中 / 无 gold 四个边界）\n")

    # ---------------- 手感对照表：recall 与 nDCG 的分工 ----------------
    # 同一份 gold（1 条）、同样命中 1 条，只改【命中位置】：
    #   recall@5 三条完全一样（位置盲），nDCG@5 相差 2.6 倍（位置敏感）
    print("同一份 gold={a}、同样命中 1 条，只改命中位置：")
    print(f"{'命中位':<8}{'recall@5':<12}{'nDCG@5':<10}{'RR':<8}")
    for pos in (1, 3, 5):
        hs = [{"chunk_id": "a" if i == pos else f"x{i}"} for i in range(1, 6)]
        print(f"#{pos:<7}{recall_at_k(hs, {'a'}, 5):<12.3f}{ndcg_at_k(hs, {'a'}, 5):<10.3f}"
              f"{reciprocal_rank(hs, {'a'}):<8.3f}")
    print("→ recall 三条相同、nDCG 逐级递减：这就是两个指标不能互相替代的证据\n")

    # ---------------- evaluate 汇总口径演示 ----------------
    rows = [
        {"id": 1, "type": "in", "gold": {"a"}, "hits": [{"chunk_id": "a"}]},          # 满分
        {"id": 2, "type": "in", "gold": {"a"}, "hits": [{"chunk_id": "x"}]},          # 全不命中
        {"id": 3, "type": "cross", "gold": {"a", "b", "c", "d"}, "hits": [{"chunk_id": "a"}]},  # 1/4
        {"id": 4, "type": "out", "gold": set(), "hits": [{"chunk_id": "x"}]},          # 无 gold → 跳过
    ]
    s = evaluate(rows)
    print("evaluate 汇总（注意 out 题被剔除，分母是 2 而不是 4 或 3）：")
    for t, g in s["by_type"].items():
        print(f"  {t:<8} n={g['n']} n_scored={g['n_scored']}  "
              f"recall@5={g['recall@5']:.3f}  mrr={g['mrr']:.3f}  ndcg@5={g['ndcg@5']:.3f}")
    print(f"  all_with_gold（仅参考）n_scored={s['all_with_gold']['n_scored']}  "
          f"recall@5={s['all_with_gold']['recall@5']:.3f}")
    print(f"  跳过无 gold 题：{s['n_no_gold']} 题")
    # in 组：满分 + 全不命中 → (1.0+0.0)/2 = 0.5；若有人把 out 题也算进去，这里会变成 0.333
    assert math.isclose(s["by_type"]["in"]["recall@5"], 0.5)
    assert s["n_no_gold"] == 1
    print("\n汇总口径断言通过（无 gold 题不进分母）")


if __name__ == "__main__":
    _self_check()