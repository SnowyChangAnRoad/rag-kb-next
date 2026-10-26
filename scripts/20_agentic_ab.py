"""实验 | W4 卡4：四组对照（A 标准 / B enhanced / C Agentic / D Agentic+cache）。

要回答的问题：把「检索几次」交给模型（C/D）与固定管线（A/B）相比，
质量、成本（LLM 调用数 / token / 费用）、延迟各差多少；给答案加一层语义缓存（D）之后，
成本能塌到什么程度、质量要付什么代价。本脚本【只落轨迹 + 基本对照表】，
recall@k 指标与文献对照（触发率 / 重复率 53% / badcase）留给卡5。

一张四组表要成立，难的不是画表，是下面五件事（1–3 同 10_hybrid_ab，4–5 为 W4 新增）：

1) 单一实验变量
   五趟共用同一个 pipeline / ranker / llm；唯一差异 = GroupParams 的四个开关 + 是否注入缓存。
   脚本里【没有一处 if group 的检索分叉】—— 检索一律走 run_agentic 内部那次
   pipeline.retrieve，其参数由 RETRIEVAL_PARAMS 与之一一对应（同一个真相源）。

2) 逐题交替，且 D2 紧跟同题 D1
   外层题、内层五趟 (A, B, C, D1, D2)。写成「外层趟、内层题」的话，后跑的趟系统性受益
   （GPU 热态 + 页面缓存），方向恰好对本实验的假设有利 —— 自欺型偏差。
   D2 必须紧跟同题 D1：D2 要命中，前提是 D1 刚把同题答案写进缓存；顺带两者共享热态。

3) 热态与确定性
   warmup（不计时、不入表）烧掉 CUDA 编译 / HNSW 首次载入；生成侧 temperature=0 + 关闭 thinking。
   「数字可信」的证据由第 4 条的不变式给出，不靠感觉。

4) 不变式（W4 新增，比 A/B 对照多的一层）
   · D1 与 C：D1 未命中 ⇒ 除 use_cache 外走【同一条链路】⇒ 产物应【逐字相同】；
   · D2 与 D1：D2 命中即【原样重放】、命中趟零模型调用 ⇒ answer 应【逐字相同】。
   这两条不是「预测」而是【可断言的事实】；实测不一致 = bug（身份漂移 / 重放失败 / 命中错记录），
   不是随机性 —— 由 check_invariants 当场校验（TODO-2）。

5) 成本口径 = 全部题目，不按 gold 抽
   token / 调用数 / 延迟都是真实流量。按「有 gold 的题」筛会让五趟题集不同，
   成本就变成第二个变量（同 10_hybrid_ab 的延迟口径纪律）。

用法（项目根 rag-kb-next 下）：
    & "<项目 venv>\Scripts\python.exe" scripts/20_agentic_ab.py --pilot   # 5 题冒烟
    & "<项目 venv>\Scripts\python.exe" scripts/20_agentic_ab.py           # 全量 20 题
    & "<项目 venv>\Scripts\python.exe" scripts/20_agentic_ab.py --noise   # 噪声地板诊断（C vs C2）
    # --noise 只跑诊断趟（同配置跑两遍，量化「什么都不改时的抖动量」），不碰主实验产物
    # 需要 GPU（嵌入 + 精排）+ 已建好的 day6_kb + 已生成 data/gold_tune.json + DEEPSEEK_API_KEY
    # pilot 分布（§0 定稿）：2 in / 1 cross / 1 partial / 1 out
跑完落盘（按批次【分文件名】，pilot 与全量互不覆盖 —— 见 prepare_run_artifacts）：
    data/w4_agentic_traces_{pilot|full}.jsonl   逐题逐趟轨迹（append）
    data/eval_w4_agentic_{pilot|full}.json      聚合对照表 + 不变式结论
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包：运行时把项目根塞进 sys.path 才能 import rag.*（同 10_hybrid_ab.py）
sys.path.insert(0, str(ROOT))

from rag.agentic import GROUPS, dump_trace, run_agentic  # noqa: E402
from rag.cache import (CACHE_JSONL, CACHE_THRESHOLD, SemanticCache,  # noqa: E402
                       build_retrieval_params)
from rag.config import (GOLD_TUNE_JSON, RECALL_K, RERANK_TOP_K, RETRIEVAL_MODE,  # noqa: E402
                        RRF_K, TESTSET_TUNE_JSONL, W4_AGENTIC_TRACE_JSONL)
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.llm import get_llm  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import BGEReranker, get_collection  # noqa: E402

# 五趟 = 组名 + 趟次。D 组两趟（D1 建缓存 / D2 原题复命）；其余组各一趟。
STAGES = (("A", 1), ("B", 1), ("C", 1), ("D", 1), ("D", 2))
EVIDENCE_JSON = ROOT / "data" / "eval_w4_agentic.json"
# 批次分文件名的模板（{scope} ∈ pilot / full）：pilot 与全量互不覆盖，
# 且文件名本身就自证「这批是什么」—— 卡5 不必靠「我记得刚跑的是全量」来判断。
TRACE_TMPL = "w4_agentic_traces_{scope}.jsonl"
EVIDENCE_TMPL = "eval_w4_agentic_{scope}.json"

# 检索参数指纹 = 缓存 namespace 的成分，也是 run_agentic → pipeline.retrieve 的实际入参。
# 必须【同一个真相源】：build_retrieval_params 的默认值就是 run_agentic 的默认入参
# （dedupe=True / top_k=None→RERANK_TOP_K / quota_cap=None / min_score=0.0 / mode=RETRIEVAL_MODE）。
# 这里显式断言一次 —— 将来谁改了 run_agentic 的默认入参，这行会当场炸，而不是静默让
# 「缓存身份」与「真实检索参数」错开。
RETRIEVAL_PARAMS = build_retrieval_params()
assert RETRIEVAL_PARAMS == {"mode": RETRIEVAL_MODE, "top_k": RERANK_TOP_K, "quota_cap": None,
                            "dedupe": True, "min_score": 0.0,
                            "RECALL_K": RECALL_K, "RERANK_TOP_K": RERANK_TOP_K, "RRF_K": RRF_K}, \
    "RETRIEVAL_PARAMS 与 run_agentic 的默认检索入参不一致 —— 缓存身份会与真实参数错开"

PILOT_PLAN = (("in", 2), ("cross", 1), ("partial", 1), ("out", 1))


def stage_label(group: str, pass_idx: int) -> str:
    """D 组两趟要能区分：D1 / D2；其余组直接是组名。"""
    return f"{group}{pass_idx}" if group == "D" else group


# ------------------------------------------------------------
# 选题（已给）
# ------------------------------------------------------------
def pick_pilot(records: list[dict]) -> list[dict]:
    """按 §0 定稿分布取各类型首题：2 in / 1 cross / 1 partial / 1 out。

    为什么按类型取首题而不是硬编码 qid：确定性可复现（不依赖手抄），且失败模式可预判
    （in 题是最容易「一轮就够」的、out 题是最容易触发退化路径的）。
    某个类型的题数不足 ⇒ 当场报错，不静默降级 —— 降级后的 pilot 分布与 §0 记录的分布不符，
    会让「pilot 通过」这个结论失效。
    """
    picked: list[dict] = []
    for qtype, n in PILOT_PLAN:
        same = [r for r in records if r["type"] == qtype]
        if len(same) < n:
            raise ValueError(f"pilot 需要 {n} 道 {qtype} 题，调参集里只有 {len(same)} 道")
        picked.extend(same[:n])
    return picked


# ------------------------------------------------------------
# 产物清理（10-19 已拍板，AI 实现）
# ------------------------------------------------------------
def prepare_run_artifacts(pilot: bool) -> tuple[Path, Path]:
    """按批次分文件名 + 每次跑前清掉缓存文件；返回 (trace_path, evidence_path)。

    三个决定与理由（TODO-1 的拍板结论）：

    1) 缓存文件（data/w4_semantic_cache.jsonl）【每次跑前删掉】。
       缓存记录带 namespace，重建实例时 _load 会按【当前 namespace】加载旧记录 ⇒
       同一份身份参数跑第二遍实验，D1 会直接命中上一轮的残留 ⇒「D1 建缓存、D2 命中」
       这个结构被打乱，实验静默失效（症状：D1 的 llm_calls 也是 0）。
       为什么不「换 namespace」保留旧数据：namespace 的成分（model_key / collection /
       variant / prompt_version / 8 项检索参数）【全是实验变量】—— 动其中任何一个都会
       污染实验。⇒ 删掉是唯一干净的做法。
    2) 轨迹 / 对照表【按批次分文件名】：pilot 与全量写同一文件会让卡5 统计的「20 题」里
       混进 pilot 的 5 题（qid 重复），对照表也会被覆盖。分文件名后文件名自证批次身份。
    3) 动作必须在【装配 cache 之前】：cache._load 在 __init__ 就跑完了，之后再删太晚。
    """
    scope = "pilot" if pilot else "full"
    trace_path = W4_AGENTIC_TRACE_JSONL.with_name(TRACE_TMPL.format(scope=scope))
    evidence_path = EVIDENCE_JSON.with_name(EVIDENCE_TMPL.format(scope=scope))
    for p in (CACHE_JSONL, trace_path, evidence_path):
        if p.exists():
            p.unlink()
            print(f"  清理上一轮产物：{p.name}")
    print(f"本次批次 = {scope} | 轨迹 → {trace_path.name} | 对照表 → {evidence_path.name}")
    return trace_path, evidence_path


# ------------------------------------------------------------
# 装配 + warmup（已给）
# ------------------------------------------------------------
def build_runner() -> RAGPipeline:
    """装配一次：collection / llm / ranker 五趟共用（省显存 + 保证环境一致）。"""
    return RAGPipeline(get_collection(), llm=get_llm(), ranker=BGEReranker())


def warmup(pipeline: RAGPipeline, question: str) -> None:
    """拿第一题跑一遍 A 组，结果丢弃、不计时、不入表。

    目的只是烧掉一次性成本：CUDA kernel 编译 / HNSW 首次载入 / LLM 首连。
    为什么用 A 组：它是最短路径（1 检索 + 1 生成），warmup 的开销最小，
    且 A 组不写缓存 —— 不会在正式跑之前污染缓存文件。
    """
    run_agentic(pipeline, question, qid="__warmup__", qtype="in", group="A")


# ------------------------------------------------------------
# 主循环（已给）：外层题、内层五趟
# ------------------------------------------------------------
def run_matrix(pipeline: RAGPipeline, records: list[dict], cache, trace_path: Path) -> list[dict]:
    """跑完 题目 × 五趟，逐条 append 轨迹到【本批次】的 trace_path，返回全部 trace 记录。

    两件事必须写对：
      · 外层题、内层趟 —— 避免「后跑的趟系统性受益」（文件头第 2 条）；
      · cache 只在 group=="D" 时注入：A/B/C 传 None，run_agentic 侧 use_cache=False
        时【零触碰缓存】；传同一个 cache 实例给 D1/D2 ⇒ D2 才能命中 D1 刚写的那条。
    轨迹路径由 prepare_run_artifacts 给出（批次分文件名），不写死默认路径 ——
    否则 pilot 与全量会 append 进同一个文件（qid 重复，卡5 分不开）。
    落盘的每条 trace 额外带两个字段（卡5 靠它们分桶）：
      · stage  —— "A" / "B" / "C" / "D1" / "D2"（D 组两趟同 group，不加这个就分不开）；
      · has_gold —— 该题是否有 gold（成本口径是全部题，但分桶统计要知道谁能算 recall）。
    """
    rows: list[dict] = []
    for r in records:
        for group, pass_idx in STAGES:
            label = stage_label(group, pass_idx)
            rec = run_agentic(pipeline, r["question"], qid=str(r["id"]), qtype=r["type"],
                              group=group, llm=pipeline.llm,
                              cache=(cache if group == "D" else None))
            rec = {**rec, "stage": label, "has_gold": bool(r["gold"])}
            dump_trace(rec, path=trace_path)
            rows.append(rec)
            print(f"{label:<3} id={str(r['id']):<3} ({r['type']:<7}) "
                  f"出口={str(rec['terminate_reason']):<20} 调用={rec['llm_calls']} "
                  f"检索={rec['retrieve_times']} 命中缓存={rec['cache_hit']} "
                  f"写缓存={rec['cache_wrote']} {rec['total_ms']:.0f}ms")
    return rows


# ------------------------------------------------------------
# 不变式校验（10-19 已拍板，AI 实现）
# ------------------------------------------------------------
def total_tokens(rec: dict) -> int:
    """本题总 token = 生成那一次（gen）+ 全部决策调用（逐轮、逐次）。

    为什么必须这样加：`gen.prompt_tokens` 只含【生成那一次】；gate / 自评 / 改写 的 token
    在 `prompt_tokens_list` 里，而它是【外层轮、内层次】的两层 list。
    ⚠️ 展平必须双重循环：`sum(rec["prompt_tokens_list"])` 会 TypeError（int + list）——
       而它长得像对的。只取 gen 会把 C 组系统性低估，而「C 组贵多少」正是预4 要检验的量。
    """
    return (rec["gen"]["prompt_tokens"] + rec["gen"]["completion_tokens"]
            + sum(t for calls in rec["prompt_tokens_list"] for t in calls))


# D1 与 C 必须逐字相同的字段。选这五个的理由：
# D1 未命中 ⇒ 除 use_cache 外走同一条链路 ⇒ 逐轮 hits / 出口 / 生成版本 / 调用次数都该一样，
# answer 更是同一次生成调用的产物。
# 不比 total_ms（墙钟，两次执行本就有调度抖动）、不比 cache_hit / cache_wrote（这两项只属 D 组，
# 由 check_invariants 的 d2_cache 明细单独核对）。
D1_EQ_C_FIELDS = ("answer", "hits_list", "terminate_reason", "prompt_variant", "llm_calls")


def check_invariants(rows: list[dict]) -> dict:
    """校验 D1≡C 与 D2≡D1，并把 D2 命中明细落盘供【人工核对】（处置口径 10-19 修订）。

    ⚠️ 与初版的关键差别（pilot 实测 id=7 触发后修订）：不再对「D2 命中率」raise。
    原因：「D2 必须 100% 命中」这个前提只在「D1 一定把本题写进缓存」时成立。
    实测 id=7（cross）是 gate 判【不检索】的题 —— D1 出口 = gate_no_retrieve，
    prompt_variant = v1-noretr，按设计【不入缓存】（该产物不依赖本库，入缓存会让模型即兴
    回答被跨题冒充成知识库答案）；0 命中路径（v1-nohit）同理。⇒ 这类题的 D2【结构上不可能
    命中】，把结构性不命中当作「实验无效」是错误前提。故命中与否【只落盘、不 raise】，
    由人工核对「D1 入了缓存的题 D2 是否命中」——但「D1 写了却 D2 没命中」这个真 bug 仍由
    人眼在 d2_cache 明细里可见，捕获力不丢。

    三条处置：
      · D2 命中明细 ⇒ 只落盘（invariants.d2_cache），不 raise。期望判据读【事实】cache_wrote，
        不从 prompt_variant 反推（反推会在准入条件一改时静默失准 = 第二个真相源）。
      · D2 ≢ D1（仅【命中趟】）⇒ raise。命中 = 原样重放 + 命中趟零模型调用（llm_calls=0）
        ⇒ 零随机性可言，差异必是 bug（重放失败 / 命中错记录）。未命中的 D2 是【重新跑整条
        链路】，与 D1 属两次独立执行（性质同 D1 vs C）⇒ 只落盘 `d2_answer_eq_d1` 供人工看。
      · D1 ≢ C ⇒ 【告警不 raise】。虽同链路，但仍是两次独立执行，API 端可能微抖；
        一次抖动就废掉整轮 ~100 次调用代价太高。但告警【必须落盘量化】，否则
        「D 组 vs C 组」的对比会被静默当成成立。
      · 三条都不许静默通过。

    raise 时把【全部】不一致项列全：重跑一整轮的代价太高，报全才能一次定位。
    """
    by_stage: dict[str, dict[str, dict]] = {}
    for rec in rows:
        by_stage.setdefault(rec["stage"], {})[rec["qid"]] = rec
    c, d1, d2 = by_stage.get("C", {}), by_stage.get("D1", {}), by_stage.get("D2", {})

    # ---- 1) D2 命中明细：只落盘，人工核对（不 raise）----
    # 每行的 d1_cache_wrote 是【事实】（D1 本趟是否真的写了缓存），不是从 prompt_variant 推的。
    # d2_answer_eq_d1：未命中的 D2 与 D1 是两次独立执行，答案是否逐字相同只作参考，不判罚。
    d2_cache = []
    for qid in sorted(d2):
        rec1, rec2 = d1.get(qid), d2[qid]
        d2_cache.append({
            "qid": qid, "qtype": rec2["qtype"],
            "d1_terminate": (rec1 or {}).get("terminate_reason"),
            "d1_cache_wrote": (rec1 or {}).get("cache_wrote"),
            "d2_cache_hit": bool(rec2["cache_hit"]),
            "d2_matched_qid": (rec2.get("cache_match") or {}).get("matched_qid"),
            "d2_answer_eq_d1": (rec2["answer"] == rec1["answer"]) if rec1 else None,
        })
    n_wrote = sum(1 for x in d2_cache if x["d1_cache_wrote"])
    n_hit = sum(1 for x in d2_cache if x["d2_cache_hit"])

    # ---- 2) D2 ≡ D1：只对【命中趟】raise（原样重放必须逐字相同）----
    mism_d2 = []
    for qid in sorted(d2):
        rec2 = d2[qid]
        if not rec2["cache_hit"]:
            continue        # 未命中的 D2 不在本条约束内 —— 它没走重放，见 d2_cache 明细
        rec1 = d1.get(qid)
        if rec1 is None:
            mism_d2.append({"qid": qid, "fields": ["D1 缺该题"]})
            continue
        bad = []
        if rec2["answer"] != rec1["answer"]:
            bad.append("answer")
        # matched_qid 必须 == 本题 qid：证明命中的是【本尊】，而不是被近义问题「借用」了
        # 别人的记录 —— 这是本轮唯一能看到的假命中信号（近义不同意图的题归 W5 构造集）。
        matched = (rec2.get("cache_match") or {}).get("matched_qid")
        if str(matched) != str(qid):
            bad.append(f"matched_qid={matched!r}")
        if bad:
            mism_d2.append({"qid": qid, "fields": bad})
    if mism_d2:
        raise ValueError(
            f"D2 ≢ D1（命中即原样重放、命中趟 llm_calls=0 ⇒ 必须逐字相同）：{mism_d2}\n"
            f"这是 bug（重放失败 / 命中错记录），不是随机性 —— 不许用随机性解释")

    # ---- 3) D1 ≡ C：逐字段比 != ，不一致只告警但必须量化落盘 ----
    mism_c = [{"qid": q, "fields": ["D1 缺该题"]} for q in sorted(set(c) - set(d1))]
    mism_c += [{"qid": q, "fields": ["C 缺该题"]} for q in sorted(set(d1) - set(c))]
    for qid in sorted(set(c) & set(d1)):
        diff = [f for f in D1_EQ_C_FIELDS if c[qid][f] != d1[qid][f]]   # != 而非 isclose
        if diff:
            mism_c.append({"qid": qid, "fields": diff})

    if mism_c:
        print(f"\n⚠️ 告警：D1 ≢ C，{len(mism_c)}/{len(c)} 题不一致（已落 invariants.d1_eq_c）")
        print("   D1 与 C 仅 use_cache 不同、走同一链路，期望【逐字相同】；不一致时 "
              "「D 组 vs C 组」的对比结论必须按此不一致率打折（但不等于实验作废）")
        for m in mism_c[:5]:
            print(f"   · id={m['qid']} 差异字段：{m['fields']}")

    # d2_eq_d1 只覆盖【命中趟】 ⇒ checked 是命中趟数（不是 D2 题数）；未命中趟的性质见 d2_cache。
    n_d2_hit = sum(1 for q in d2 if d2[q]["cache_hit"])
    return {
        "d2_cache": {
            "checked": len(d2), "n_d1_wrote": n_wrote, "n_d2_hit": n_hit,
            "rows": d2_cache,
            "note": ("命中率只落盘、人工核对（不 raise）。D2 未命中可能是 gate_no_retrieve / "
                     "0 命中路径 —— 这两条按设计不入缓存，D2 结构上不可能命中；"
                     "只需核对「D1 写了缓存（d1_cache_wrote=True）的题 D2 是否都命中」"),
        },
        "d1_eq_c": {"checked": len(c), "mismatched": mism_c, "ok": not mism_c},
        "d2_eq_d1": {"checked": n_d2_hit, "mismatched": mism_d2, "ok": not mism_d2},
        "passed": not mism_d2 and not mism_c,
    }


# ------------------------------------------------------------
# 噪声地板（10-19 新增，诊断趟，不进五趟对照表）
# ------------------------------------------------------------
NOISE_TMPL = "eval_w4_noise_floor_{scope}.json"
NOISE_JSON = ROOT / "data" / "eval_w4_noise_floor.json"


def run_noise_floor(pipeline: RAGPipeline, records: list[dict]) -> dict:
    """同一配置（C 组参数）跑两趟，量化「什么都不改时的抖动量」（噪声地板）。

    为什么必须单列这一趟：全量实测 `D1 ≢ C = 18/20`，而 D1 与 C 只差一个 `use_cache` 开关、
    走同一条链路 —— 这 18/20 里既有「缓存引入的差异」（按设计应为 0）也有「API 端非确定性」。
    不给噪声地板，「D1 比 C 多 0.10 次调用」这类结论无法解读：它可能整段都落在噪声里。
    C2 = C 的第二次**独立执行** ⇒ C vs C2 的差异就是【纯噪声】，作为 D1 vs C 的参照下界。

    口径与主实验一致：外层题、内层两趟（C, C2），避免「后跑的趟系统性受益」。
    **不碰缓存文件、不删主实验产物**（诊断趟只读不写），落盘独立文件。
    比较字段沿用 D1_EQ_C_FIELDS —— 它比的就是「两条同配置链路」的产物，与本趟同义
    （另抄一份字段清单就是第二个真相源）。
    """
    pairs: list[dict] = []
    for r in records:
        recs = []
        for label in ("C", "C2"):
            rec = run_agentic(pipeline, r["question"], qid=str(r["id"]), qtype=r["type"],
                              group="C", llm=pipeline.llm)
            recs.append(rec)
            print(f"{label:<4} id={str(r['id']):<3} ({r['type']:<7}) "
                  f"出口={str(rec['terminate_reason']):<20} 调用={rec['llm_calls']} "
                  f"检索={rec['retrieve_times']} {rec['total_ms']:.0f}ms")
        a, b = recs
        pairs.append({
            "qid": a["qid"], "qtype": a["qtype"],
            "fields": [f for f in D1_EQ_C_FIELDS if a[f] != b[f]],
            "terminate": [a["terminate_reason"], b["terminate_reason"]],
            "llm_calls": [a["llm_calls"], b["llm_calls"]],
            "retrieve_times": [a["retrieve_times"], b["retrieve_times"]],
        })
    n = len(pairs)
    bad = [p for p in pairs if p["fields"]]
    field_counts: dict[str, int] = {}
    for p in bad:
        for f in p["fields"]:
            field_counts[f] = field_counts.get(f, 0) + 1
    return {"n": n, "mismatched": len(bad),
            "mismatch_rate": (len(bad) / n if n else 0.0),
            "field_counts": field_counts, "rows": pairs}


def print_noise(res: dict, pilot: bool) -> None:
    """打印噪声地板 + 逐题差异清单。"""
    print("\n--- 噪声地板（C vs C2：同配置、两次独立执行）---")
    print(f"规模：{'pilot' if pilot else '全量 20 题'} | 不一致 {res['mismatched']}/{res['n']} "
          f"= {res['mismatch_rate']:.2f} | 字段分布：{res['field_counts']}")
    print("   用法：D1 ≢ C 的不一致率【减去】本条，才是缓存可能引入的差异 —— 本条是参照下界")
    for p in res["rows"]:
        if p["fields"]:
            print(f"   · id={p['qid']} ({p['qtype']}) 差异字段={p['fields']} "
                  f"出口={p['terminate']} 调用={p['llm_calls']}")


# ------------------------------------------------------------
# 聚合（10-19 已拍板：总额比）
# ------------------------------------------------------------
# 参与倍数计算的列。cache_hits 不入倍数（计数，不是流量）。
RATIO_KEYS = ("llm_calls", "tokens", "cost", "total_ms", "retrieve_times")


def aggregate(rows: list[dict]) -> dict:
    """把逐条 trace 聚成对照表：每题均值（绝对量）+ 总额，倍数用【总额比】。

    口径（10-19 拍板）：
      · **总额比** = sum(趟) / sum(A)。理由：成本是【账单概念】—— 20 题跑完总共花多少
        就是这趟的真实开销；极端情况（某题爆到 6 次调用）是真实开销的一部分，用中位数
        会被抹掉。预4 的「倍数」本来也要求按 llm_calls 落盘值重算总额。
      · **同时报绝对量**（每题均值）：只报倍数看不出量级 —— D2 延迟「×0.02」到底是 40ms
        还是 400ms，必须靠绝对量（预5 的「一次 embedding 10–50ms」是绝对口径）。
      · 全部题参与，不按 has_gold 抽（成本口径纪律：筛题会让五趟的题集不同）。

    ⚠️ token 必须用 total_tokens(rec)：gen 只含生成那一次，逐轮决策调用的 token 在
       prompt_tokens_list（两层 list）。只取 gen 会把 C/D1 系统性低估。
    """
    order = [stage_label(g, i) for g, i in STAGES]
    grouped: dict[str, list[dict]] = {}
    for rec in rows:
        grouped.setdefault(rec["stage"], []).append(rec)

    totals: dict[str, dict] = {}
    by_stage: dict[str, dict] = {}
    for label in order:
        recs = grouped.get(label, [])
        if not recs:
            continue
        n = len(recs)
        tot = {
            "llm_calls": sum(rec["llm_calls"] for rec in recs),
            "tokens": sum(total_tokens(rec) for rec in recs),
            "cost": sum(rec["gen"]["cost"] for rec in recs),
            "total_ms": sum(rec["total_ms"] for rec in recs),
            "retrieve_times": sum(rec["retrieve_times"] for rec in recs),
        }
        totals[label] = tot
        reasons: dict[str, int] = {}
        for rec in recs:
            reasons[rec["terminate_reason"]] = reasons.get(rec["terminate_reason"], 0) + 1
        by_stage[label] = {
            "n": n,
            **{k: tot[k] / n for k in RATIO_KEYS},   # 每题均值 = 绝对量
            "cache_hits": sum(1 for rec in recs if rec["cache_hit"]),
            "totals": tot,                            # 总额 = 倍数的分子
            "terminate_reasons": reasons,             # 卡5 核对降级路径占比
        }

    base = totals.get("A")
    if base is None:
        raise ValueError("aggregate: 缺 A 组 —— 倍数的基准没了，数字无法解读")
    ratios = {label: {k: (tot[k] / base[k] if base[k] else float("nan"))
                      for k in RATIO_KEYS}
              for label, tot in totals.items()}

    return {"by_stage": by_stage, "ratios_vs_A": ratios,
            "method": "总额比 sum(趟)/sum(A)"}


# ------------------------------------------------------------
# 打印（已给）
# ------------------------------------------------------------
def print_table(agg: dict, inv: dict, pilot: bool) -> None:
    """已给：打印对照表 + 不变式结论。"""
    print(f"\n参数：RETRIEVAL_MODE={RETRIEVAL_MODE} RECALL_K={RECALL_K} RRF_K={RRF_K} "
          f"RERANK_TOP_K={RERANK_TOP_K} | 缓存阈值={CACHE_THRESHOLD}")
    stage_names = ", ".join(stage_label(g, i) for g, i in STAGES)
    print(f"规模：{'pilot 5 题' if pilot else '全量 20 题'} | 五趟：{stage_names} "
          f"| 倍数口径：{agg.get('method')}")

    cols = ("llm_calls", "tokens", "cost", "total_ms", "retrieve_times", "cache_hits")
    print(f"\n{'趟':<5}{'题数':>5}" + "".join(f"{c:>14}" for c in cols))
    for label, stat in agg["by_stage"].items():
        cells = "".join(f"{stat.get(c, float('nan')):>14.2f}" for c in cols)
        print(f"{label:<5}{stat.get('n', 0):>5}{cells}")

    print(f"\n{'vs A':<8}" + "".join(f"{c:>14}" for c in cols[:-1]))
    for label, ratio in agg["ratios_vs_A"].items():
        cells = "".join(f"{ratio.get(c, float('nan')):>14.2f}" for c in cols[:-1])
        print(f"{label:<8}{cells}")

    print("\n--- 不变式（D2 命中率只落盘、人工核对，不 raise）---")
    dc = inv["d2_cache"]
    print(f"D1 入缓存 {dc['n_d1_wrote']}/{dc['checked']} 题 | D2 命中 {dc['n_d2_hit']}/{dc['checked']} 题"
          f" —— 核对「D1 写了缓存的题 D2 是否都命中」：")
    print(f"   {'id':<4} | {'类型':<7} | {'D1 出口':<19} | {'D1写了':<6} | {'D2命中':<6} | "
          f"{'matched_qid':<11} | answer==D1")
    for x in dc["rows"]:
        print(f"   {str(x['qid']):<4} | {str(x['qtype']):<7} | {str(x['d1_terminate']):<19} | "
              f"{str(x['d1_cache_wrote']):<6} | {str(x['d2_cache_hit']):<6} | "
              f"{str(x['d2_matched_qid']):<11} | {x['d2_answer_eq_d1']}")
    ok_c, ok_d2 = inv["d1_eq_c"], inv["d2_eq_d1"]
    print(f"D1 ≡ C  ：{'通过' if ok_c['ok'] else '不一致 ' + str(ok_c['mismatched'])}")
    print(f"D2 ≡ D1（仅命中趟 {ok_d2['checked']} 题）："
          f"{'通过' if ok_d2['ok'] else '不一致 ' + str(ok_d2['mismatched'])}")
    print(f"不变式总判定：{'通过' if inv['passed'] else '未通过'}")


# ------------------------------------------------------------
# 落盘（已给）
# ------------------------------------------------------------
def dump_evidence(rows: list[dict], agg: dict, inv: dict, pilot: bool, evidence_path: Path) -> None:
    """已给：落盘对照表 + 逐题明细 + 指纹到【本批次】的 evidence_path（pilot / full 各一份）。

    轨迹全量在本批次的 traces jsonl；本文件放【聚合结论 + 一份轻量逐题表】，
    让卡5 不必重新解析整个 jsonl 就能画表。指纹（testset / gold sha256）用于对外报数时
    自证「当时跑的是哪份数据」；`stage_scope` 与 `trace_file` 自证批次。
    """
    per_question = [
        {"qid": r["qid"], "qtype": r["qtype"], "stage": r["stage"],
         "terminate_reason": r["terminate_reason"], "llm_calls": r["llm_calls"],
         "retrieve_times": r["retrieve_times"], "cache_hit": r["cache_hit"],
         # cache_wrote 落进逐题表：人工核对「D1 写了缓存、D2 是否命中」只读这里即可
         "cache_wrote": r["cache_wrote"],
         "total_tokens": total_tokens(r),
         "cost": r["gen"]["cost"], "total_ms": r["total_ms"],
         "prompt_variant": r["prompt_variant"],
         "fused_ids": [h["chunk_id"] for h in r["fused_hits"]],
         "answer_preview": r["answer"][:100]}
        for r in rows
    ]
    payload = {
        "experiment": "w4_agentic_ab",
        "stage_scope": "pilot(5)" if pilot else "full(20)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "RETRIEVAL_MODE": RETRIEVAL_MODE, "RECALL_K": RECALL_K, "RRF_K": RRF_K,
            "RERANK_TOP_K": RERANK_TOP_K, "cache_threshold": CACHE_THRESHOLD,
            "retrieval_params": RETRIEVAL_PARAMS,
            "groups": {g: {"max_rounds": p.max_rounds, "use_gate": p.use_gate,
                           "use_rewrite": p.use_rewrite, "use_cache": p.use_cache}
                       for g, p in GROUPS.items()},
        },
        "fingerprints": {
            "testset": TESTSET_TUNE_JSONL.name, "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
            "gold": GOLD_TUNE_JSON.name, "gold_sha256": sha256_file(GOLD_TUNE_JSON),
            "trace_file": TRACE_TMPL.format(scope="pilot" if pilot else "full"),
        },
        "aggregate": agg,
        "invariants": inv,
        "per_question": per_question,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{evidence_path}")


# ------------------------------------------------------------
# 主入口（已给）
# ------------------------------------------------------------
def main() -> None:
    """已给：装配 → warmup → 跑矩阵 → 聚合/校验 → 打印 → 落盘。

    加 --noise 时改走诊断分支：只跑噪声地板（C vs C2），不删主实验产物、不建缓存。
    """
    args = sys.argv[1:]
    pilot = "--pilot" in args
    records = load_gold()["records"]
    if pilot:
        records = pick_pilot(records)
        print(f"pilot 选出 {len(records)} 题："
              f"{[(str(r['id']), r['type']) for r in records]}")

    if "--noise" in args:
        pipeline = build_runner()
        warmup(pipeline, records[0]["question"])
        res = run_noise_floor(pipeline, records)
        print_noise(res, pilot)
        noise_path = NOISE_JSON.with_name(
            NOISE_TMPL.format(scope="pilot" if pilot else "full"))
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        with open(noise_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": "w4_noise_floor",
                       "scope": "pilot(5)" if pilot else "full(20)",
                       "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "params": {"RETRIEVAL_MODE": RETRIEVAL_MODE, "RECALL_K": RECALL_K,
                                  "RRF_K": RRF_K, "RERANK_TOP_K": RERANK_TOP_K},
                       "result": res}, f, ensure_ascii=False, indent=2)
        print(f"\n已落盘：{noise_path}")
        return

    trace_path, evidence_path = prepare_run_artifacts(pilot)
    pipeline = build_runner()
    cache = SemanticCache(retrieval_params=RETRIEVAL_PARAMS, threshold=CACHE_THRESHOLD,
                          path=CACHE_JSONL)
    warmup(pipeline, records[0]["question"])

    rows = run_matrix(pipeline, records, cache, trace_path)
    agg = aggregate(rows)                          # TODO-3
    inv = check_invariants(rows)                   # TODO-2
    print_table(agg, inv, pilot)
    dump_evidence(rows, agg, inv, pilot, evidence_path)


if __name__ == "__main__":
    main()
