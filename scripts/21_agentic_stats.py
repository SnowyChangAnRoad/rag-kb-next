"""实验 | W4 卡5（分析卡）：全量轨迹统计 + 文献对照 + badcase 定位。

要回答的问题（第十四条预注册五条 + 两项通关口径）：
    预1  C 组重检索触发率 < 10% ？（**报两个分母**，对照文献 ~10%）
    预2  「重检索结果与上一轮 top-k 完全相同」> 53%？（对照文献 53%）
    预3  增益方向（**主读 cite_at_k**；正确率因 in/out 饱和仅作辅助列）
    预4  成本倍数 A=1 / B=1.05–1.15 / C=2.5–3.0 / D2≈0.05；延迟 C ×3–5
    预5  D2 token < 5% C、延迟 ≈ 一次 embedding(10–50ms)、命中率 ≈100%
    通关 ① 我的触发率 vs 文献 10%（含「其中多少与上一轮完全相同」对照 53%）
         ③ ≥1 个「标准 RAG 答错 / Agentic 答对」实例（预期 cross）
    另：gate 判「不检索」但该题有 gold 的题（本批 id=7/11）按**方案 A** 单列报 —— 实测属
    「通用原理题不检索也能答好」的**生产/实验口径错位**，不并进约束违反口径（报告 §4.1）。

本脚本【只读】卡4 的两份产物（`data/w4_agentic_traces_{scope}.jsonl` +
`data/eval_w4_agentic_{scope}.json`），**不重跑检索、不重跑生成** —— 轨迹是唯一真相源，
重跑一次会把 D1≡C 的噪声地板结论一起废掉（设计要点 §11.6）。分两层：

    零 API 层（默认，可复现）：触发率 / 重复率 / 出口分布 / recall@k·MRR·nDCG /
        正则引用覆盖率 / D2 延迟与相似度 / 成本与延迟对照
    裁判锚点层（`--judge`）：复用 rag.eval 的 judge_correct + judge_citations，
        只跑 A 与 C 两组（≈80 次裁判调用），用途有二：
          ① 验证「零 API 代理」的可信度（同一批记录上并排给 judge 口径与正则口径）；
          ② 给出 A/C **逐题逐字对照**，筛「A 错 / C 对」（通关问题 3）

用法（项目根 rag-kb-next 下）：
    & "<项目 venv>\\Scripts\\python.exe" scripts/21_agentic_stats.py            # 零 API，全量
    & "<项目 venv>\\Scripts\\python.exe" scripts/21_agentic_stats.py --judge    # 加 A/C 裁判锚点
    & "<项目 venv>\\Scripts\\python.exe" scripts/21_agentic_stats.py --pilot    # 换批次
跑完落盘：data/eval_w4_agentic_stats_{full|pilot}.json

------------------------------------------------------------
口径声明（读表前必须知道，写进报告的「局限」一节）
------------------------------------------------------------
1) **排序口径不可比**：A 组按 `rerank_score` 序；C/D 组按跨轮 RRF 融合序（设计要点 §11.3
   「不重排」）。所以 C 组 recall@k 的升降**不能**单独归因于「质量」—— in 组的「单块跨越
   top-3 边界」变化属口径差异（W2 第十三条教训）。本表因此**分题型**报，不给混合 headline。
2) **重复率只能在 top-k 口径算**：trace 每轮只落 `hits_list`（= 送进融合池的 top-3），
   没有全召回池 ⇒ 「重检索结果与上一轮相同」= top-3 集合/次序相同，与文献的「结果相同」
   不是同一个数（文献口径未公开到该粒度）—— 对照时必须声明。
3) **gate 那次调用的 token 不可恢复**：`gate_rec` 只落 `need_retrieval/parsed/raw`，
   `decide_retrieve` 返回的 `prompt_tokens` 没有被 `run_agentic` 收下（B 组无 gate，不受影响）。
   ⇒ C/D 的 token 被系统性低估。按本题平均 token 粗算，gate 提示词 ≈ 常数 system + 问题
   ≈ 数百 token，占 C 组每题 ~2200 token 的个位数百分比，方向为**低估**，量级须在报告里声明。
4) **cross/partial 是来源级 gold**（`unit="source"`，分母 = 文件数），in 是页级
   （分母 = 页内块数）⇒ 两者数值不同量纲，本表按题型 macro，绝不可加总。
5) **D2 是混合均值**：18 题命中（只花一次编码 + 查表）+ 2 题结构性未命中
   （`gate_no_retrieve`，按设计不入缓存 ⇒ D2 重跑整条链路）。任何 D2 的均值都必须把
   这两类分开报，否则「延迟塌到一次 embedding」会被那 2 题的 ~2.5s 淹掉。
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包：运行时把项目根塞进 sys.path 才能 import rag.*（同 20_agentic_ab.py）
sys.path.insert(0, str(ROOT))

from rag.config import (GOLD_TUNE_JSON, PRICE_IN_PER_M, PRICE_OUT_PER_M,  # noqa: E402
                        RECALL_K, RERANK_TOP_K, RETRIEVAL_MODE, RRF_K,
                        W4_AGENTIC_TRACE_JSONL)
from rag.eval import count_valid_cited, extract_cited_ids  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.metrics import (EVAL_KS, SOURCE_UNIT, as_source_hits, evaluate)  # noqa: E402

STAGES = ("A", "B", "C", "D1", "D2")
TRACE_TMPL = "w4_agentic_traces_{scope}.jsonl"
EVIDENCE_TMPL = "eval_w4_agentic_{scope}.json"
STATS_TMPL = "eval_w4_agentic_stats_{scope}.json"

# 文献对照系（第十四条修订10）：只作对照，不是目标值。
LITERATURE = {
    "retrieval_trigger_rate": 0.10,      # 「仅 ~10% 的案例选择重检索」
    "repeat_same_as_last_round": 0.53,   # 「其中 53% 重检索结果与上一轮完全相同」
    "cost_max_multiple": 3.3,            # 「成本最高 3.3×」
}

# 与 20_agentic_ab.py 同源：倍数按【总额比】算（账单概念），同时报每题均值（绝对量）。
RATIO_KEYS = ("llm_calls", "retrieve_times", "tokens", "cost", "total_ms")


# ------------------------------------------------------------
# 载入（已给）
# ------------------------------------------------------------
def load_traces(path: Path) -> list[dict]:
    """读本批次的轨迹 jsonl。缺文件直接报错并指出该跑哪条命令 —— 不许静默空跑。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到轨迹文件 {path.name} —— 先跑 "
            f"scripts/20_agentic_ab.py{'' if 'full' in path.name else ' --pilot'}")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def index_by_stage(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """{stage: {qid: rec}}。D 组两趟同 group ⇒ 必须按 `stage` 分桶（见 20_agentic_ab 的说明）。"""
    by_stage: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], {})[r["qid"]] = r
    return by_stage


def load_meta() -> tuple[list[dict], dict[str, dict]]:
    """题目元信息（type / unit / gold / question）来自 gold_tune.json —— 与卡4 同一份真相源。

    为什么不能只从 trace 读：trace 里有 qtype，但**没有 gold、没有 unit、也没有 question 原文**
    （trace 只有 qid + qtype）。裁判需要 question 原文，指标需要 gold 与 unit ⇒ 只能回 gold_tune。
    """
    payload = load_gold()
    records = payload["records"]
    return records, {str(r["id"]): r for r in records}


# ------------------------------------------------------------
# 记账口径（已给，10-19 拍板的两处修正）
# ------------------------------------------------------------
def decision_prompt_tokens(rec: dict) -> int:
    """自评 + 改写的 prompt token（`prompt_tokens_list` 是【外层轮、内层次】两层 list）。

    ⚠️ 展平必须双重循环 —— `sum(rec["prompt_tokens_list"])` 会 TypeError（int + list），
    而它长得像对的。只取 gen 会把 C 组系统性低估，而「C 组贵多少」正是预4 要检验的量。
    ⚠️ **不含 gate**：gate 的 prompt_tokens 没进 trace（见文件头口径 3）—— 这是已知低估。
    """
    return sum(t for calls in rec["prompt_tokens_list"] for t in calls)


def full_cost(rec: dict) -> float:
    """修正后的成本口径：生成段 + 自评/改写的 token 折价。

    为什么必须修：卡4 的 `aggregate` 用的是 `rec["gen"]["cost"]` —— **只含生成段**。
    决策调用（自评 / 改写）每次都吃了输入 token，却不进成本列 ⇒ C/D 组成本被低估，
    而「C 组贵多少」正是预4 的核心量。价格常量从 config 读，不许手抄。
    （gate 那一次的 token 仍缺，见口径 3；方向同为低估。）
    """
    in_tok = rec["gen"]["prompt_tokens"] + decision_prompt_tokens(rec)
    out_tok = rec["gen"]["completion_tokens"]
    return in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M


def total_tokens(rec: dict) -> int:
    """本题总 token = 生成那次 + 全部决策调用（不含 gate，见口径 3）。"""
    return (rec["gen"]["prompt_tokens"] + rec["gen"]["completion_tokens"]
            + decision_prompt_tokens(rec))


def gold_hit_count(fused: list[dict], gold: set[str], unit: str | None) -> int:
    """融合池对 gold 的命中数 —— **按该题的 gold 粒度投影**后再取交集。

    投影这一步不能省：cross/partial 的 gold 是**文件名**（`unit="source"`），
    而 fused_hits 的 `chunk_id` 是块 id ⇒ 不投影时交集**恒为空**，会把这题误判成
    「零命中」。复用 metrics 的 `as_source_hits`（只留每个来源第一次出现的位置）——
    另写一份投影就是第二个真相源（10_hybrid_ab / metrics 已吃过这个亏）。
    """
    if not gold:
        return 0
    projected = as_source_hits(fused) if unit == SOURCE_UNIT else fused
    return len({h["chunk_id"] for h in projected} & gold)


def regex_cite_slots(answer: str, n_slots: int) -> tuple[list[int], float | None]:
    """零 API 的「引用覆盖率」近似：模型引用的编号里，落在 1..len(fused_hits) 的个数 / len(fused_hits)。

    与 `rag.eval` 的 `cite_at_k` 差在**一处**：这里不做裁判的「该编号是否真的支撑那句话」判定
    （`count_valid_cited` 的第一个参数传空 invalid）⇒ 口径偏**高**。
    这处差异不是缺陷而是**待测量**：`--judge` 会在同一批记录上并排给两个口径，
    差异大小就是「这个代理能不能用」的证据。
    """
    if n_slots <= 0:
        return [], None
    cited = extract_cited_ids(answer)
    valid = count_valid_cited(cited, [], n_slots)
    return valid, len(valid) / n_slots


# ------------------------------------------------------------
# TODO-1｜触发率与重复率（预1 / 预2，卡5 的 headline）
# ------------------------------------------------------------
def trigger_and_repeat(by_stage: dict[str, dict[str, dict]],
                       stages: tuple[str, ...] = STAGES) -> dict:
    """TODO-1：算重检索触发率（双分母）、相邻轮重复率、改写逐字相同率。

    输出（键名固定，打印与落盘都按它读）：
      {
        "trigger": {stage: {"n_all": int, "n_triggered": int, "rate_all": float|None,
                            "n_retrieval": int, "rate_retrieval": float|None}},
        "repeat":  {stage: {"adjacent_pairs": int, "same_set": int, "same_order": int,
                            "rate_same_set": float|None, "rate_same_order": float|None}},
        "rewrite": {stage: {"n_rewrites": int, "n_identical": int, "rate": float|None}},
      }

    契约（每一条都对应一个会静默出错的写法）：
      · **触发 = `retrieve_times >= 2`** —— 不是「出口是 cap」，也不是「自评判不足」：
        判据必须落在【真的又检索了一次】这个事实上（出口只是结果，不是次数）。
      · **两个分母**（预1 明确要求）：
          rate_all       = n_triggered / 全部题数（= len(by_stage[stage])）
          rate_retrieval = n_triggered / 走了检索的题数（`retrieve_times >= 1`）
        第二个分母把 `gate_no_retrieve`（0 次检索）的题排除 —— 文献的 10% 是「在**能**重检索的
        场景里有多少选了重检索」，两个数不并排报就没法和文献对齐。
      · **相邻轮**：对每题 `hits_list` 取 (i-1, i) 这一对对，**不是**末轮 vs 融合池
        （那是 `fused_vs_last_round_topk_diff` 的口径，对单轮题恒真，在这里没有分辨率）。
      · **相同两档**：`same_set` = 两轮 chunk_id 集合相等（位置盲）；`same_order` = 列表逐位相等。
        ⚠️ 两轮 hits 已是「本轮 rerank 后的 top-3」⇒ 这是 **top-k 口径**（见文件头口径 2），
        报告里必须说明它比文献的「结果相同」更严。
      · **改写逐字相同**：读 `query_repeat_flags[1:]`（第 0 位恒 False，是占位）除以
        `len(question_list) - 1`。这一项与「检索结果相同」是两回事（改了词也可能落回同一批块），
        两列并报才能解释预2 为什么可能翻车。
      · 分母为 0 ⇒ 该 rate 记 **None**（打印成 "-"），不许静默写 0.0 —— 0.0 会被读成
        「一个都没发生」，而事实是「什么都没发生，无从谈起」。

    黑盒验收（实现完先跑这三条，都能手算）：
      · A 组：全部 1 轮 ⇒ trigger 三项全 0、adjacent_pairs = 0
      · B 组：全部 2 轮 ⇒ rate_all = rate_retrieval = 1.0、adjacent_pairs = 20、n_rewrites = 20
      · C 组（全量实测）⇒ rate_all = 0.35、rate_retrieval ≈ 0.389、adjacent_pairs = 10、same_set = 2
    """
    trigger: dict[str, dict] = {}
    repeat: dict[str, dict] = {}
    rewrite: dict[str, dict] = {}

    for st in stages:
        recs = by_stage.get(st, {})
        n_all = len(recs)
        # 触发 = 真的又检索了一次。不读 terminate_reason：出口只是结果，不是次数。
        n_triggered = sum(1 for r in recs.values() if r["retrieve_times"] >= 2)
        n_retrieval = sum(1 for r in recs.values() if r["retrieve_times"] >= 1)
        trigger[st] = {
            "n_all": n_all, "n_triggered": n_triggered,
            # 分母为 0 ⇒ None（打印成 "-"）。写 0.0 会被读成「一次都没触发」，与「无从谈起」不同。
            "rate_all": (n_triggered / n_all) if n_all else None,
            "n_retrieval": n_retrieval,
            "rate_retrieval": (n_triggered / n_retrieval) if n_retrieval else None,
        }

        pairs = same_set = same_order = 0
        for r in recs.values():
            hl = r["hits_list"]                       # 每轮已是「本轮 rerank 后的 top-k」
            for i in range(1, len(hl)):               # 只比【相邻轮】，不是末轮 vs 融合池
                a = [h["chunk_id"] for h in hl[i - 1]]
                b = [h["chunk_id"] for h in hl[i]]
                pairs += 1
                same_set += bool(set(a) == set(b))
                same_order += bool(a == b)
        repeat[st] = {
            "adjacent_pairs": pairs, "same_set": same_set, "same_order": same_order,
            "rate_same_set": (same_set / pairs) if pairs else None,
            "rate_same_order": (same_order / pairs) if pairs else None,
        }

        n_rewrites = sum(max(0, len(r["question_list"]) - 1) for r in recs.values())
        # 第 0 位恒 False 是占位（原始 query 没有「上一轮」）⇒ 必须切片 [1:]
        n_identical = sum(sum(1 for f in r["query_repeat_flags"][1:] if f)
                          for r in recs.values())
        rewrite[st] = {
            "n_rewrites": n_rewrites, "n_identical": n_identical,
            "rate": (n_identical / n_rewrites) if n_rewrites else None,
        }

    return {"trigger": trigger, "repeat": repeat, "rewrite": rewrite}


# ------------------------------------------------------------
# TODO-2｜约束违反 badcase 候选（预1 声明，通关问题与报告都要用）
# ------------------------------------------------------------
def violation_candidates(by_stage: dict[str, dict[str, dict]],
                         meta: dict[str, dict]) -> list[dict]:
    """TODO-2：筛出「自评说证据够、但证据其实根本不相关」的题 —— 预1 声明的约束违反 badcase。

    为什么要它：预1 写死了「若出现『模型改用参数知识补答而不重查』，那不是正常表现，
    而是**约束违反 badcase，须单列**」。本项目 prompt 的硬约束是「只能依据资料、不编造、
    资料无关才拒答」⇒ 自评判 `sufficient` 却没有任何 gold 证据进池，
    意味着模型在**无相关知识库证据**的情况下照样答了 —— 正是那条声明要抓的东西。

    判据（三条同时成立才算候选）：
      1) `terminate_reason == "self_eval_sufficient"` —— 自评明确说「够了」（不重查）
      2) 该题**有 gold**（out 题无 gold，不算违反）
      3) `gold_hit_count(fused_hits, gold, unit) == 0` —— 融合池里一条 gold 都没有

    输出：每题一项
      {"qid","qtype","unit","stage","exit","n_fused","n_gold_hit","answer_preview":"前 200 字"}

    ⚠️ 两条必须写对：
      · **按 unit 投影**（用 `gold_hit_count`）—— 直接拿 chunk_id 与 gold 求交集的话，
        cross/partial 的 gold 是文件名，交集**恒空** ⇒ 这两组会被**整组误报**成违反。
        这是本函数最容易静默出错的一处（错法还很"合理"：数字看起来有值）。
      · 这只是**候选**，不是结论：证据不相关时模型也可能正确拒答（那不算违反）。
        所以必须带 `answer_preview` 落盘，由人读答案再判 —— 脚本不替人做这个判断。

    黑盒验收：
      · 结果里**不应出现任何 out 题**（无 gold 被剔除）
      · 结果里**不应出现 cross/partial 的全量误报**（投影正确时这两组只有极少数真候选）
      · 用构造数据（人为把 fused_hits 换成空列表）跑一遍，该题必须进候选 —— 验证判据不是恒假式
    """
    cands: list[dict] = []
    for st in STAGES:
        recs = by_stage.get(st, {})
        for qid, rec in sorted(recs.items(), key=lambda kv: int(kv[0])):
            # 判据 1：自评明确说「够了」（不重查）。A/B 组没有自评，天然不会进来。
            if rec["terminate_reason"] != "self_eval_sufficient":
                continue
            q = meta.get(qid)
            if q is None:
                continue
            # 判据 2：out 题无 gold，不适用「证据不足却作答」这条 —— 拒答题本来就该无 gold。
            if not q["gold"]:
                continue
            # 判据 3：融合池对该题 gold 零命中 ⇒ 自评通过建立在「不相关证据」上。
            # 必须按 unit 投影（gold_hit_count 内部做），否则 cross/partial 会整组误报。
            n_hit = gold_hit_count(rec["fused_hits"], q["gold"], q.get("unit"))
            if n_hit != 0:
                continue
            cands.append({
                "qid": qid, "qtype": q["type"], "unit": q.get("unit"), "stage": st,
                "exit": rec["terminate_reason"],
                "n_fused": len(rec["fused_hits"]), "n_gold_hit": n_hit,
                "n_gold": len(q["gold"]),
                "answer_preview": rec["answer"].replace("\n", " ")[:200],
            })
    return cands


# ------------------------------------------------------------
# 方案 A｜gate 不检索但有 gold 的题（单列，不吃进 violations 口径）
# ------------------------------------------------------------
def gate_no_evidence_candidates(by_stage: dict[str, dict[str, dict]],
                                meta: dict[str, dict],
                                judge: dict | None = None) -> dict:
    """按「方案 A」单列的诊断：**gate 判了不检索，但这道题有 gold**。

    为什么不并进 `violation_candidates`：两者抓的是**不同层**的失败，混在一起就说不清了
      · `violation_candidates` = 自评层放行（`self_eval_sufficient`）且池内零 gold
      · 本清单 = gate 层直接短路（`gate_no_retrieve`），检索压根没发生 ⇒ 自评根本没机会说话

    ⚠️ 本批实测结论（10-21）：id=7/11 两题是**通用原理题**，**不检索也能答好**
       （id=11 的 C 组答案比 A 组更完整）。gate 判「不检索」在生产中合理，评测因无引用
       判错 ⇒ 属**生产/实验口径错位**，**不是 gate 缺陷**。清单保留作候选，但语义是
       「口径错位候选」；是否为 badcase 必须人工读 `answer_preview`，不由脚本定性。

    判据（两条，全是事实字段，零 API）：
      1) `terminate_reason == "gate_no_retrieve"`
      2) 该题有 gold（out 题无 gold ⇒ 判不检索是**合理**行为，不计入本清单）

    ⚠️ 「答案有没有拒答」**不在脚本里判**：关键词法在 out 题上会误命中（judge.py 文件头已有
       这个教训）。有裁判结果时（`judge` 传了 C 组逐题表）并排给 `correct` / `reason`，
       其余字段交给人工读 `answer_preview` —— 脚本只负责把候选摆齐。

    输出：
      {"n_unique_qid", "n_confirmed_wrong_in_C", "by_stage": {stage: 条数},
       "rows": [{qid, qtype, unit, n_gold, stages:[...], c_correct, c_reason,
                 answer_preview}]}
    同一题在 C / D1 / D2 三趟都会复现（D2 是结构性未命中），按 qid 去重、`stages` 合并，
    否则同一道题会被数成 3 条，读者会误以为有 6 个 badcase。
    """
    per_qid: dict[str, dict] = {}
    stage_counts: dict[str, int] = {}
    c_rows = ((judge or {}).get("per_question") or {}).get("C") or []
    c_verdict = {r["qid"]: r for r in c_rows}

    for st in STAGES:
        recs = by_stage.get(st, {})
        stage_counts[st] = 0
        for qid, rec in sorted(recs.items(), key=lambda kv: int(kv[0])):
            if rec["terminate_reason"] != "gate_no_retrieve":
                continue
            q = meta.get(qid)
            if q is None or not q["gold"]:     # out 题（无 gold）判不检索是正确的
                continue
            stage_counts[st] += 1
            row = per_qid.get(qid)
            if row is None:
                cv = c_verdict.get(qid)
                row = per_qid[qid] = {
                    "qid": qid, "qtype": q["type"], "unit": q.get("unit"),
                    "n_gold": len(q["gold"]), "stages": [],
                    "c_correct": cv["correct"] if cv else None,
                    "c_reason": cv["correct_reason"] if cv else None,
                    "answer_preview": rec["answer"].replace("\n", " ")[:200],
                }
            row["stages"].append(st)

    rows = [per_qid[k] for k in sorted(per_qid, key=int)]
    return {
        "n_unique_qid": len(rows),
        "n_confirmed_wrong_in_C": sum(1 for r in rows if r["c_correct"] is False),
        "by_stage": stage_counts,
        "rows": rows,
    }


# ------------------------------------------------------------
# TODO-3｜裁判锚点 A vs C（预3 主读列 cite_at_k + 通关问题 3）
# ------------------------------------------------------------
def judge_anchor(by_stage: dict[str, dict[str, dict]], meta: dict[str, dict],
                 stages: tuple[str, ...] = ("A", "C")) -> dict:
    """TODO-3：对 A / C 两组跑裁判（复用 rag.eval 的两条链路），产出正确率与 cite_at_k。

    为什么只跑 A 与 C 而不跑全五趟：**裁判的用途是两个，都不需要 B/D**
      ① 预3 主读列的增益方向 = A vs C（B 是加固组，D≈C；D2 与 D1 答案相同，判两次反而是第二个口径）
      ② 通关问题 3 找「标准 RAG 答错 / Agentic 答对」= A 对 C 的逐题对照，样本就是 A/C

    每题两跳（各 1 次调用 ≈ 1~1.5k token 输入），20 题 × 2 组 × 2 = **80 次调用**。
    ⚠️ 输入取 `rec["fused_hits"]`（trace 已存全字段含 text / source / page）与
       `meta[qid]["question"]` —— **不许**重新检索或重新生成，一次都不许（见文件头）。

    输出：
      {
        "groups": {stage: {"n", "answer_acc", "correct",
                           "cite_at_k_judge", "cite_at_k_regex", "slot_valid", "slot_total",
                           "zero_cite"}},
        "per_question": {stage: [{qid, qtype, correct, correct_reason,
                                  cite_total, cite_valid, invalid, cited_ids, regex_slots}]},
        "a_wrong_c_right": [qid...],       # 通关问题 3 的候选实例
        "judge_tokens": {"prompt": int, "completion": int},
      }

    口径要点：
      · `cite_at_k_judge` = Σ len(valid_cited_ids) / Σ top_k（**pooled**，同 rag.eval.summarize）：
        valid_cited_ids = `count_valid_cited(extract_cited_ids(answer), jcit["invalid"], len(fused_hits))`
        —— 分母钉在「送进 prompt 的条数」，不是模型写了几条（旧口径的分母是模型自由度）。
      · `cite_at_k_regex` = 零 API 代理，在**同一批记录**上并排算（`regex_cite_slots`）：
        两列的差就是代理的系统偏差，写进报告当可信度证据。
      · out 题的 `cite_at_k` 口径：拒答题若没写引用 ⇒ `slot_total` 那一题的分子分母都是 0。
        预3 明确「out 组若被 gate 判不需检索而无引用，**不计入 cite_at_k 分母**」⇒
        汇总时须把 out 且零引用 的题剔除后再 pooled，并单独报 `zero_cite` 题数。
      · 裁判自身有噪声（`temperature=0` 也不保证逐字复现），结论须与噪声地板一起读。

    黑盒验收：
      · 打印 A 与 C 的逐题对照（正确 / 引用覆盖率），人工能一眼看出方向
      · `a_wrong_c_right` 必须逐个带理由与 answer 预览，供人复核（不许只给题号）
      · 全零引用题（两个分子分母都为 0）不许当 0 分混进平均值 —— 用 pooled 口径天然规避，
        但汇总代码里要有 `zero_cite` 计数落盘，否则「分母被悄悄缩小」没人发现
    """
    # scripts/ 不是包 ⇒ 用绝对导入（相对导入在这里会 ImportError）
    from rag.judge import judge_citations, judge_correct  # 局部 import：零 API 路径不依赖裁判层

    groups: dict[str, dict] = {}
    per_question: dict[str, list] = {}
    tokens = {"prompt": 0, "completion": 0}

    for st in stages:
        recs = by_stage.get(st, {})
        rows: list[dict] = []
        correct = 0
        slot_valid = slot_total = 0
        reg_valid = reg_total = 0
        zero_cite = out_excluded = 0

        for qid, rec in sorted(recs.items(), key=lambda kv: int(kv[0])):
            q = meta[qid]
            fused, answer = rec["fused_hits"], rec["answer"]
            n_slots = len(fused)

            jc = judge_correct(q["question"], fused, answer)
            jcit = judge_citations(q["question"], fused, answer)
            for u in (jc["usage"], jcit["usage"]):
                tokens["prompt"] += u["prompt_tokens"]
                tokens["completion"] += u["completion_tokens"]

            cited = extract_cited_ids(answer)
            valid_ids = count_valid_cited(cited, jcit["invalid"], n_slots)
            reg_ids, _ = regex_cite_slots(answer, n_slots)

            # 预3 口径：out 且零引用 ⇒ 两个口径**同时**剔出分母（否则 A 组被系统性压低）
            excluded = q["type"] == "out" and not valid_ids and not reg_ids
            if excluded:
                out_excluded += 1
            else:
                slot_valid += len(valid_ids)
                slot_total += n_slots
                reg_valid += len(reg_ids)
                reg_total += n_slots
                if n_slots and not valid_ids:
                    zero_cite += 1

            correct += bool(jc["correct"])
            rows.append({
                "qid": qid, "qtype": q["type"],
                "correct": bool(jc["correct"]), "correct_reason": jc["reason"],
                "cite_total": jcit["total"], "cite_valid": jcit["valid"],
                "invalid": jcit["invalid"], "cited_ids": cited,
                "judge_valid_ids": valid_ids, "regex_valid_ids": reg_ids,
                "n_slots": n_slots,
                "answer_preview": answer.replace("\n", " ")[:200],
            })
            print(f"    {st} id={qid:<3}({q['type']:<7}) "
                  f"{'✅' if jc['correct'] else '❌'} 引用 {jcit['valid']}/{jcit['total']} "
                  f"槽位 {len(valid_ids)}/{n_slots} | {jc['reason'][:30]}")

        n = len(rows)
        groups[st] = {
            "n": n, "correct": correct,
            "answer_acc": (correct / n) if n else None,
            "cite_at_k_judge": (slot_valid / slot_total) if slot_total else None,
            "cite_at_k_regex": (reg_valid / reg_total) if reg_total else None,
            "slot_valid": slot_valid, "slot_total": slot_total,
            "zero_cite": zero_cite, "out_no_cite_excluded": out_excluded,
        }
        per_question[st] = rows

    # 通关问题 3：A 错 / C 对 —— 逐题对照，带理由（人复核用）
    a_wrong_c_right: list[dict] = []
    if "A" in per_question and "C" in per_question:
        a_rows = {r["qid"]: r for r in per_question["A"]}
        for c in per_question["C"]:
            a = a_rows.get(c["qid"])
            if a and not a["correct"] and c["correct"]:
                a_wrong_c_right.append({
                    "qid": c["qid"], "qtype": c["qtype"],
                    "a_reason": a["correct_reason"], "c_reason": c["correct_reason"],
                    "a_answer": a["answer_preview"], "c_answer": c["answer_preview"],
                })

    return {"groups": groups, "per_question": per_question,
            "a_wrong_c_right": a_wrong_c_right, "judge_tokens": tokens}


# ------------------------------------------------------------
# 以下为已给：出口分布 / 检索指标 / D2 / 成本延迟 / 打印 / 落盘
# ------------------------------------------------------------
def exits(by_stage: dict[str, dict[str, dict]]) -> dict:
    """每条趟的出口分布 + 逐题一行（卡5 报告要引用「哪 8 道题触发了重检索」）。"""
    out = {}
    for st in STAGES:
        recs = by_stage.get(st, {})
        out[st] = {
            "counts": dict(Counter(r["terminate_reason"] for r in recs.values())),
            "rows": [{"qid": q, "qtype": r["qtype"], "retrieve_times": r["retrieve_times"],
                      "exit": r["terminate_reason"], "llm_calls": r["llm_calls"]}
                     for q, r in sorted(recs.items(), key=lambda kv: int(kv[0]))],
        }
    return out


def retrieval_metrics(by_stage: dict[str, dict[str, dict]], records: list[dict]) -> dict:
    """五趟的 recall@k / MRR / nDCG@k（分题型 macro）。

    读 `rec["fused_hits"]` —— 那是【送进生成段的最终序】（A 组 rerank 序、C/D 组 RRF 融合序）。
    ⚠️ 融合池序与 A 组序**不同源**（设计要点 §11.3）⇒ 本表只作「同题型内横比 + 声明」，
       不能当质量结论单独使用；in 组的变化须用 recall@5/@10 交叉验证。
    D2 命中趟的 `fused_hits` 是**首趟重放**的同一份（缓存命中），所以 D2 与 D1 的检索层指标
    结构上必然相同 —— 这不是"发现"，落盘只为完整。
    """
    out = {}
    for st in STAGES:
        rows = []
        for q in records:
            rec = by_stage.get(st, {}).get(str(q["id"]))
            if rec is None:
                raise KeyError(f"阶段 {st} 缺 id={q['id']} 的轨迹 —— 批次不完整，指标无意义")
            rows.append({"id": q["id"], "type": q["type"], "unit": q["unit"],
                         "gold": q["gold"], "hits": rec["fused_hits"]})
        out[st] = evaluate(rows, ks=EVAL_KS)
    return out


def d2_stats(by_stage: dict[str, dict[str, dict]], evidence: dict | None) -> dict:
    """D2 的命中率与**命中趟专属**延迟（预5）。

    为什么命中趟必须单独算：D2 是混合均值 —— 18 题命中（encode + 查表）+ 2 题结构性未命中
    （`gate_no_retrieve` ⇒ 不入缓存 ⇒ D2 重跑整条 ~2.5s 链路）。混在一起算，
    「延迟塌到一次 embedding」会被那 2 题淹掉（~273ms 均值 vs 命中趟实际 ~7.5ms）。

    两份产物的交叉校验：`cache_wrote` / `cache_hit` 的事实**在 trace 里**
    （卡4 落盘），这里与 `eval_w4_agentic_{scope}.json` 的 `invariants.d2_cache` 对账；
    不一致就 raise —— 两份真相源打架时不许挑一个信。
    """
    d2 = by_stage.get("D2", {})
    hits_rows = [r for r in d2.values() if r["cache_hit"]]
    miss_rows = [r for r in d2.values() if not r["cache_hit"]]

    def _stat(rs, key):
        vals = [r[key] for r in rs]
        return {"mean": statistics.fmean(vals), "med": statistics.median(vals),
                "min": min(vals), "max": max(vals)} if vals else None

    ms = [r["total_ms"] for r in hits_rows]
    sims = [r["cache_match"]["similarity"] for r in hits_rows]
    enc = [r["cache_match"]["encode_ms"] for r in hits_rows]
    lk = [r["cache_match"]["lookup_ms"] for r in hits_rows]

    n_wrote = sum(1 for r in by_stage.get("D1", {}).values() if r["cache_wrote"])
    res = {
        "n": len(d2), "n_hit": len(hits_rows), "n_miss": len(miss_rows),
        "n_d1_wrote": n_wrote,
        # 真正的判据：**D1 写了缓存的题** D2 是否都命中（结构性不命中的题不在其中）
        "hit_rate_all": len(hits_rows) / len(d2) if d2 else None,
        "hit_rate_among_wrote": len(hits_rows) / n_wrote if n_wrote else None,
        "hit_total_ms": _stat(hits_rows, "total_ms"),
        "hit_encode_ms_mean": statistics.fmean(enc) if enc else None,
        "hit_lookup_ms_mean": statistics.fmean(lk) if lk else None,
        "similarity_min": min(sims) if sims else None,
        "similarity_max": max(sims) if sims else None,
        "miss_structural": [
            {"qid": r["qid"], "qtype": r["qtype"], "exit": r["terminate_reason"],
             "d1_exit": by_stage["D1"][r["qid"]]["terminate_reason"],
             "d1_cache_wrote": by_stage["D1"][r["qid"]]["cache_wrote"],
             "d2_total_ms": r["total_ms"]}
            for r in sorted(miss_rows, key=lambda x: int(x["qid"]))
        ],
        "hit_total_ms_raw": ms,   # 直接落原始值，报告要重算分位数时不必再解析 jsonl
    }
    if evidence is not None:
        inv = (evidence.get("invariants") or {}).get("d2_cache") or {}
        if inv:
            if inv.get("n_d1_wrote") != n_wrote or inv.get("n_d2_hit") != len(hits_rows):
                raise ValueError(
                    f"两份产物不一致：trace 侧 D1 写了 {n_wrote} / D2 命中 {len(hits_rows)}，"
                    f"对照表侧写了 {inv.get('n_d1_wrote')} / 命中 {inv.get('n_d2_hit')} "
                    f"—— 两份真相源打架，先查是谁算错了，不许挑一个信")
            res["cross_check"] = "与 eval_w4_agentic_*.json 的 invariants.d2_cache 一致"
    return res


def cost_latency(by_stage: dict[str, dict[str, dict]]) -> dict:
    """成本与延迟对照（预4）：**总额比** sum(趟)/sum(A)，同时报每题均值（绝对量）。

    两处口径与卡4 的 aggregate 不同，都是本轮拍板的修正：
      · `cost` 用 **full_cost**（生成段 + 自评/改写 token 折价）—— 卡4 只算 gen.cost；
      · `tokens` 用 **total_tokens**（展平两层 list）—— 与卡4 一致。
    两者都仍缺 gate 那一次调用（口径 3），方向为低估，报告须声明。
    """
    out: dict[str, dict] = {}
    for st in STAGES:
        recs = list(by_stage.get(st, {}).values())
        n = len(recs)
        tot = {
            "llm_calls": sum(r["llm_calls"] for r in recs),
            "retrieve_times": sum(r["retrieve_times"] for r in recs),
            "tokens": sum(total_tokens(r) for r in recs),
            "cost": sum(full_cost(r) for r in recs),
            "total_ms": sum(r["total_ms"] for r in recs),
        }
        out[st] = {"n": n,
                   "per_question": {k: tot[k] / n for k in RATIO_KEYS},
                   "totals": tot}
    base = out["A"]["totals"]
    ratios = {st: {k: (v / base[k] if base[k] else None) for k, v in d["totals"].items()}
              for st, d in out.items()}
    return {"by_stage": out, "ratios_vs_A": ratios, "method": "总额比 sum(趟)/sum(A)"}


def cite_regex_all(by_stage: dict[str, dict[str, dict]], records: list[dict]) -> dict:
    """五趟的**零 API** 引用覆盖率（正则代理）—— 与裁判口径并排看的参照列。"""
    out = {}
    for st in STAGES:
        slot_valid, slot_total, zero_cite, n_no_gold = 0, 0, 0, 0
        for q in records:
            rec = by_stage.get(st, {}).get(str(q["id"]))
            if rec is None:
                continue
            n_slots = len(rec["fused_hits"])
            valid, _ = regex_cite_slots(rec["answer"], n_slots)
            if q["type"] == "out" and not valid:
                n_no_gold += 1          # out 且零引用：预3 声明不计入 cite_at_k 分母
                continue
            slot_valid += len(valid)
            slot_total += n_slots
            if n_slots and not valid:
                zero_cite += 1
        out[st] = {"slot_valid": slot_valid, "slot_total": slot_total,
                   "cite_at_k_regex": (slot_valid / slot_total) if slot_total else None,
                   "zero_cite": zero_cite, "out_no_cite_excluded": n_no_gold}
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _num(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def print_trigger_repeat(tr: dict) -> None:
    """已给：预1 / 预2 的对照表（含文献列）。"""
    print("\n--- 预1 重检索触发率（触发 = retrieve_times ≥ 2）---")
    print(f"{'趟':<5}{'题数':>5}{'触发':>5}{'/全体':>9}{'/走了检索的题':>14}{'走了检索':>9}")
    for st, d in tr["trigger"].items():
        print(f"{st:<5}{d['n_all']:>5}{d['n_triggered']:>5}"
              f"{_pct(d['rate_all']):>9}{_pct(d['rate_retrieval']):>14}{d['n_retrieval']:>9}")
    print(f"   文献对照系：~{_pct(LITERATURE['retrieval_trigger_rate'])}（仅 ~10% 的案例选择重检索）")

    print("\n--- 预2 重检索结果与上一轮 top-3 完全相同（相邻轮对）---")
    print(f"{'趟':<5}{'相邻轮对':>9}{'集合相同':>9}{'次序相同':>9}"
          f"{'率(集合)':>10}{'率(次序)':>10}{'改写逐字相同':>13}")
    for st, d in tr["repeat"].items():
        rw = tr["rewrite"][st]
        rw_txt = f"{rw['n_identical']}/{rw['n_rewrites']}={_pct(rw['rate'])}"
        print(f"{st:<5}{d['adjacent_pairs']:>9}{d['same_set']:>9}{d['same_order']:>9}"
              f"{_pct(d['rate_same_set']):>10}{_pct(d['rate_same_order']):>10}{rw_txt:>13}")
    print(f"   文献对照系：{_pct(LITERATURE['repeat_same_as_last_round'])}（重检索结果与上一轮完全相同）")
    print(f"   改写逐字相同列 = query_repeat_flags 为 True 的次数 / 改写次数；"
          f"⚠️ 与「结果相同」是两回事，两列并报才能解释预2")


def print_exits(ex: dict) -> None:
    """已给：出口分布 + 「哪几道题触发了重检索」。"""
    print("\n--- 出口分布（terminate_reason）---")
    print(f"{'趟':<5}" + "".join(f"{k:>20}" for k in
                                 ("self_eval_sufficient", "cap", "gate_no_retrieve",
                                  "query_repeat", "cache_hit")))
    for st, d in ex.items():
        c = d["counts"]
        print(f"{st:<5}" + "".join(f"{c.get(k, 0):>20}" for k in
                                   ("self_eval_sufficient", "cap", "gate_no_retrieve",
                                    "query_repeat", "cache_hit")))
    for st in ("C", "D1"):
        trig = [r["qid"] for r in ex[st]["rows"] if r["retrieve_times"] >= 2]
        print(f"   {st} 触发重检索的题：{trig}（类型："
              f"{[next(r['qtype'] for r in ex[st]['rows'] if r['qid'] == q) for q in trig]}）")


def print_metrics(rm: dict, cite: dict) -> None:
    """已给：检索层指标 + 正则引用覆盖率的对照表（分题型 macro）。"""
    print(f"\n--- 检索层指标（fused_hits，未截断前全量；A=rerank 序 / C,D=RRF 融合序，口径不可比）---")
    print(f"{'趟':<5}{'题型':<9}{'n':>3}{'recall@3':>10}{'recall@5':>10}{'recall@10':>10}"
          f"{'mrr':>8}{'ndcg@5':>9}")
    for st in STAGES:
        for t, g in rm[st]["by_type"].items():
            print(f"{st:<5}{t:<9}{g['n']:>3}{g['recall@3']:>10.3f}{g['recall@5']:>10.3f}"
                  f"{g['recall@10']:>10.3f}{g['mrr']:>8.3f}{g['ndcg@5']:>9.3f}")
    print("\n--- 引用覆盖率（零 API 正则代理；分母 = 送进 prompt 的条数）---")
    print(f"{'趟':<5}{'有效引用/槽位':>16}{'覆盖率':>10}{'零引用题':>10}{'out无引用已剔':>14}")
    for st, d in cite.items():
        slot_txt = f"{d['slot_valid']}/{d['slot_total']}"
        print(f"{st:<5}{slot_txt:>16}{_pct(d['cite_at_k_regex']):>10}"
              f"{d['zero_cite']:>10}{d['out_no_cite_excluded']:>14}")


def print_cost(cl: dict) -> None:
    """已给：成本/延迟对照（总额比 + 每题均值），含文献上限 3.3×。"""
    print(f"\n--- 成本与延迟（{cl['method']}；cost = 修正口径，含自评/改写 token 折价）---")
    cols = ("llm_calls", "retrieve_times", "tokens", "cost", "total_ms")
    # cost 的量级是 1e-3 元 ⇒ 必须给 6 位小数，否则 0.0012 会被打成 "0.00"（看起来像免费）
    fmt = {"llm_calls": ">16.2f", "retrieve_times": ">16.2f", "tokens": ">16.1f",
           "cost": ">16.6f", "total_ms": ">16.1f"}
    print(f"{'趟':<5}{'题数':>5}" + "".join(f"{c + '(每题)':>16}" for c in cols))
    for st, d in cl["by_stage"].items():
        print(f"{st:<5}{d['n']:>5}" + "".join(f"{d['per_question'][c]:{fmt[c]}}" for c in cols))
    print(f"\n{'vs A':<6}" + "".join(f"{c:>12}" for c in cols))
    for st, r in cl["ratios_vs_A"].items():
        print(f"{st:<6}" + "".join(f"{_num(r[c]):>12}" for c in cols))
    print(f"   文献对照系：成本最高 {LITERATURE['cost_max_multiple']}×"
          f"（预4 预测：B=1.05–1.15 / C=2.5–3.0 / D2≈0.05）")


def print_d2(d: dict) -> None:
    """已给：预5 的三项（命中率 / 命中趟延迟 / 相似度）。"""
    print("\n--- 预5 D2（缓存命中趟）---")
    print(f"命中 {d['n_hit']}/{d['n']} 题 | D1 写了缓存的 {d['n_d1_wrote']} 题 ⇒ "
          f"命中率（写了缓存的题里）={_pct(d['hit_rate_among_wrote'])}"
          f"（全体 {_pct(d['hit_rate_all'])}）")
    h = d["hit_total_ms"]
    if h:
        print(f"命中趟 total_ms：均值 {h['mean']:.1f} / 中位 {h['med']:.1f} / "
              f"范围 {h['min']:.1f}–{h['max']:.1f}（预测区间 10–50ms）")
        print(f"  其中 encode {d['hit_encode_ms_mean']:.1f}ms + lookup {d['hit_lookup_ms_mean']:.2f}ms"
              f" | 相似度 {d['similarity_min']:.4f}–{d['similarity_max']:.4f}")
    for m in d["miss_structural"]:
        print(f"  结构性未命中 id={m['qid']}（{m['qtype']}）：D1 出口={m['d1_exit']}、"
              f"D1 写了缓存={m['d1_cache_wrote']} ⇒ 按设计不入缓存，D2 只能重跑整链路 "
              f"({m['d2_total_ms']:.0f}ms)")
    if "cross_check" in d:
        print(f"  {d['cross_check']}")


def print_violations(cands: list[dict]) -> None:
    """已给：约束违反候选（人工复核清单，脚本不下结论）。"""
    print(f"\n--- 约束违反 badcase 候选（自评判 sufficient 但融合池零 gold 命中）：{len(cands)} 条 ---")
    for c in cands:
        print(f"   id={c['qid']} ({c['qtype']}/{c['unit']}) {c['stage']} 出口={c['exit']} "
              f"池内 {c['n_fused']} 条 / gold 命中 {c['n_gold_hit']}")
        print(f"      答案前 200 字：{c['answer_preview']}")
    if not cands:
        print("   无 —— 说明自评通过时融合池至少含 1 条 gold（本轮无该失败模式）")


def print_gate_no_evidence(d: dict) -> None:
    """方案 A 的单列小节：gate 判不检索、但该题有 gold（口径错位候选，非 gate 缺陷）。"""
    print(f"\n--- 口径错位候选：gate 判不检索但该题有 gold（方案 A 单列；不计入约束违反）："
          f"{d['n_unique_qid']} 道题 ---")
    print(f"   按趟出现次数：{d['by_stage']}（同一题跨趟复现只算 1 道）")
    for r in d["rows"]:
        cv = "未跑裁判" if r["c_correct"] is None else \
            ("C 判对" if r["c_correct"] else f"C 判错（{r['c_reason']}）")
        print(f"   id={r['qid']}（{r['qtype']}/{r['unit']}，gold {r['n_gold']} 个）"
              f" 出现在 {'/'.join(r['stages'])} | {cv}")
        print(f"      答案前 200 字：{r['answer_preview']}")
    if not d["rows"]:
        print("   无 —— gate 没有把有 gold 的题判成不检索")
    elif d["n_confirmed_wrong_in_C"]:
        print(f"   其中 C 组裁判判错 {d['n_confirmed_wrong_in_C']} 道 ⇒ "
              f"按引用口径判错；实测答案内容正确 ⇒ 属评测口径错位，不是自评放行、也不是 gate 故障")


def print_judge(judge: dict) -> None:
    """已给：裁判锚点的逐题对照（A vs C）+ 「A 错 / C 对」实例明细。"""
    print("\n--- 裁判锚点（A vs C，人工复核清单）---")
    print(f"{'趟':<4}{'正确':>6}{'答案正确率':>10}{'cite_at_k(裁判)':>16}"
          f"{'cite_at_k(正则)':>16}{'零引用题':>9}{'out无引用已剔':>14}")
    for st, g in judge["groups"].items():
        slot = f"{g['slot_valid']}/{g['slot_total']}={_pct(g['cite_at_k_judge'])}"
        print(f"{st:<4}{g['correct']:>4}/{g['n']:<2}{_pct(g['answer_acc']):>10}{slot:>16}"
              f"{_pct(g['cite_at_k_regex']):>16}{g['zero_cite']:>9}"
              f"{g['out_no_cite_excluded']:>14}")
    jt = judge["judge_tokens"]
    print(f"   裁判开销：输入 {jt['prompt']} token / 输出 {jt['completion']} token")

    print(f"\n   A 错 / C 对（通关问题 3 的候选实例）：{len(judge['a_wrong_c_right'])} 条")
    for x in judge["a_wrong_c_right"]:
        print(f"     id={x['qid']}（{x['qtype']}）")
        print(f"       A 判错理由：{x['a_reason']}")
        print(f"       C 判对理由：{x['c_reason']}")
        print(f"       A 答案：{x['a_answer']}")
        print(f"       C 答案：{x['c_answer']}")


def print_scorecard(tr: dict, cl: dict, d2: dict, rm: dict, cite: dict,
                    judge: dict | None) -> None:
    """已给：五条预注册预测的逐条对照（预测 / 实测 / 判定）。判定只按数字，不替人下解释。"""
    def verdict(ok: bool | None) -> str:
        return "未判" if ok is None else ("命中" if ok else "打脸")

    c = tr["trigger"]["C"]
    rep = tr["repeat"]["C"]
    b, cc, d1, d2r = (cl["ratios_vs_A"][s] for s in ("B", "C", "D1", "D2"))
    c_cite = cite["C"]["cite_at_k_regex"] if cite else None
    a_cite = cite["A"]["cite_at_k_regex"] if cite else None

    rows = [
        ("预1 触发率 < 10%",
         f"C {_pct(c['rate_all'])}（全体）/ {_pct(c['rate_retrieval'])}（走了检索的题）；"
         f"文献 ~10%",
         verdict(c["rate_all"] is not None and c["rate_all"] < LITERATURE["retrieval_trigger_rate"])),
        ("预2 与上一轮 top-3 完全相同 > 53%",
         f"C {_pct(rep['rate_same_set'])}（集合）/{_pct(rep['rate_same_order'])}（次序），"
         f"{rep['adjacent_pairs']} 对；文献 53%",
         verdict(rep["rate_same_set"] is not None
                 and rep["rate_same_set"] > LITERATURE["repeat_same_as_last_round"])),
        ("预3 增益方向（主读 cite_at_k）",
         f"零 API 代理：A {_pct(a_cite)} → C {_pct(c_cite)}；"
         f"裁判口径见 --judge 输出",
         verdict(None)),
        ("预4 成本倍数",
         f"B {_num(b['llm_calls'])}（预测 1.05–1.15）/ C {_num(cc['llm_calls'])}"
         f"（预测 2.5–3.0）/ D2 {_num(d2r['llm_calls'])}（预测 ≈0.05）；"
         f"文献上限 3.3×",
         verdict(None)),
        ("预5 D2 token < 5% C / 延迟 10–50ms / 命中率 ≈100%",
         f"token {_num(d2r['tokens'])} | 命中趟 {d2['hit_total_ms']['mean']:.1f}ms"
         f"（区间 10–50ms）| 写了缓存的题命中 {_pct(d2['hit_rate_among_wrote'])}",
         verdict(d2r["tokens"] is not None and d2r["tokens"] < 0.05)),
    ]
    print("\n=== 预注册对照（第十四条，预测已冻结；判定只按数字）===")
    for name, actual, vd in rows:
        print(f"  [{vd:^4}] {name}")
        print(f"          实测：{actual}")
    if judge:
        print("\n  预3 的裁判口径（A vs C）：")
        for st in judge["groups"]:
            g = judge["groups"][st]
            print(f"    {st}: 答案正确率 {_pct(g['answer_acc'])}（{g['correct']}/{g['n']}）"
                  f" | cite_at_k(裁判) {_pct(g['cite_at_k_judge'])}"
                  f" | cite_at_k(正则代理) {_pct(g['cite_at_k_regex'])}"
                  f" | 零引用题 {g['zero_cite']}")
        print(f"    A 错 / C 对：{[x['qid'] for x in judge['a_wrong_c_right']] or '无'}"
              f"（明细见上方「裁判锚点」）")


def dump_stats(payload: dict, path: Path) -> None:
    """已给：落盘统计结论 + 指纹（报告对外报数时自证「跑的是哪份数据」）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{path}")


def main() -> None:
    """已给：读产物 → 零 API 统计 →（可选）裁判锚点 → 打印 → 落盘。"""
    args = sys.argv[1:]
    pilot = "--pilot" in args
    want_judge = "--judge" in args
    scope = "pilot" if pilot else "full"

    trace_path = W4_AGENTIC_TRACE_JSONL.with_name(TRACE_TMPL.format(scope=scope))
    evidence_path = W4_AGENTIC_TRACE_JSONL.with_name(EVIDENCE_TMPL.format(scope=scope))
    out_path = W4_AGENTIC_TRACE_JSONL.with_name(STATS_TMPL.format(scope=scope))

    rows = load_traces(trace_path)
    by_stage = index_by_stage(rows)
    records, meta = load_meta()
    missing = [st for st in STAGES if len(by_stage.get(st, {})) != len(records)]
    if missing:
        raise ValueError(f"批次不完整：{missing} 的题数 ≠ {len(records)} —— 卡5 的统计无意义，"
                         f"先补跑 scripts/20_agentic_ab.py")
    evidence = json.load(open(evidence_path, encoding="utf-8")) if evidence_path.exists() else None

    print(f"批次 = {scope} | 轨迹 {trace_path.name}（{len(rows)} 条）| 题数 {len(records)}")

    tr = trigger_and_repeat(by_stage)                                    # TODO-1
    ex = exits(by_stage)
    rm = retrieval_metrics(by_stage, records)
    cite = cite_regex_all(by_stage, records)
    cl = cost_latency(by_stage)
    d2 = d2_stats(by_stage, evidence)
    viol = violation_candidates(by_stage, meta)                          # TODO-2

    judge = judge_anchor(by_stage, meta) if want_judge else None          # TODO-3
    gate_ev = gate_no_evidence_candidates(by_stage, meta, judge)          # 方案 A 单列

    print_trigger_repeat(tr)
    print_exits(ex)
    print_metrics(rm, cite)
    print_cost(cl)
    print_d2(d2)
    print_violations(viol)
    print_gate_no_evidence(gate_ev)
    if judge:
        print_judge(judge)
    print_scorecard(tr, cl, d2, rm, cite, judge)

    payload = {
        "experiment": "w4_agentic_stats",
        "scope": f"{scope}({len(records)})",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"RETRIEVAL_MODE": RETRIEVAL_MODE, "RECALL_K": RECALL_K, "RRF_K": RRF_K,
                   "RERANK_TOP_K": RERANK_TOP_K, "ks": list(EVAL_KS)},
        "fingerprints": {"trace_file": trace_path.name,
                         "trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
                         "gold": GOLD_TUNE_JSON.name, "gold_sha256": sha256_file(GOLD_TUNE_JSON)},
        "literature": LITERATURE,
        "trigger_repeat": tr,
        "exits": ex,
        "retrieval_metrics": rm,
        "cite_regex": cite,
        "cost_latency": cl,
        "d2": d2,
        "violations": viol,
        "gate_no_evidence": gate_ev,
        "judge_anchor": judge,
    }
    dump_stats(payload, out_path)


if __name__ == "__main__":
    main()
