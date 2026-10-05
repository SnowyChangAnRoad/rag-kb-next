"""实验 | W1 卡9：验收「接进 pipeline 的召回结果」==「卡7/卡8 的脚本口径」。

卡9 改了什么
    config.RETRIEVAL_MODE 开关 + pipeline._retrieve 的召回段换成 hybrid_retrieve + 
    answer/answer_stream 透传 mode + /health 报 retrieval_mode。四处都是接线，
    接线最容易出的错是「参数没透传下去」—— 而它的表现是【所有题都跑 dense】，
    指标看上去正常，只是 hybrid 组悄悄等于 dense 组。所以必须有一个能当场暴露它的检查。

本卡检什么
    对 16 道有 gold 的题 × 2 个 mode，把【真实 RAGPipeline 跑出来的 top-k】与
    【卡8 用卡7 落盘证据重建的 top-k】逐位比对。三方交叉验证：
        ① 卡7 落盘证据（当时的召回结果）      ② 卡8 的重建逻辑（零 GPU 复现）
        ③ 现在 pipeline 的新鲜召回结果
    三者对上，才说明「服务路径 == 实验脚本口径」；任何一处接线错、调用顺序错、
    后处理参数不一致，都会在某一题上逐位不等。

    为什么【走 answer() 而不是 _retrieve()】：卡9 的改动有一半在 answer()
    （mode 透传、trace 记录、out["retrieval_mode"]），只调 _retrieve() 就绕过了那半，
    接线错了也测不出来。代价是生成段会被调到 —— 所以塞一个假 LLM：不花钱、不联网、
    不引入采样噪声（本卡真正被检的是检索段，生成段的输出不参与任何断言）。

    为什么复用卡8 的 rebuild_top_k 而不是抄一份：它是「卡7 落盘证据 → 配额后 top-3」
    的唯一解释。抄一份就有两个版本，一旦漂移，本卡会拿错误的期望值去判 pipeline 对错 ——
    那比不检更糟。

用法（项目根 rag-kb-next 下，需要 GPU 加载 reranker，秒级）：
    & "<项目 venv>\\Scripts\\python.exe" scripts/12_pipeline_mode_check.py
    前置：卡7 跑过（data/eval_w1_ab.json 存在）+ gold_tune.json 未被改动
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import RERANK_TOP_K, RETRIEVAL_MODE, SOURCE_CAP  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retriever import BGEReranker, get_collection  # noqa: E402

EVIDENCE_JSON = ROOT / "data" / "eval_w1_ab.json"
MODES = ("dense", "hybrid")


# ------------------------------------------------------------
# TODO-1：走公共接口取一次召回结果
# ------------------------------------------------------------
def retrieve_via_pipeline(pipeline: RAGPipeline, question: str, mode: str) -> list[str]:
    """TODO-1：用 pipeline.answer() 跑一次，返回送进 prompt 的 top-k 的 chunk_id 列表。

    要求：
      a) 调 pipeline.answer(...)，并把这四个参数与卡7 脚本对齐 ——
         它们不是本卡的自由变量，只要有一个不一致，比对就会「全错」而看不出原因：
             dedupe=True, top_k=RERANK_TOP_K, quota_cap=SOURCE_CAP, mode=mode
      b) ⚠️ 先断言 out["retrieval_mode"] == mode 再取 hits。
         这条断言就是「参数有没有真的透传下去」的唯一探针：若 answer() 忘了把 mode
         交给 _retrieve()，真实跑的是 dense，而这一跑【看起来完全正常】——
         只有断言能拦住它。断言消息里把 out["retrieval_mode"] 打出来。
      c) 返回 [h["chunk_id"] for h in out["hits"]]（hits 已按 select_diverse 排好序，
         顺序就是装配顺序，别重排）
    """
    out = pipeline.answer(question=question, dedupe=True, top_k=RERANK_TOP_K,
                          quota_cap=SOURCE_CAP, mode=mode)
    # ⚠️ 先断言、再取 hits：若 answer() 忘了把 mode 交给 _retrieve()，真实跑的是 dense，
    #    而这一跑【看起来完全正常】—— 只有这条断言能拦住它。
    if out["retrieval_mode"] != mode:
        raise AssertionError(f'参数透传失败：要求 mode={mode}，'
                             f'实际 out["retrieval_mode"]={out["retrieval_mode"]}')
    return [h["chunk_id"] for h in out["hits"]]


# ------------------------------------------------------------
# TODO-2：逐题比对
# ------------------------------------------------------------
def verify_mode(pipeline: RAGPipeline, records: dict, expected: dict, mode: str) -> list[str]:
    """TODO-2：把一个 mode 的 16 道题逐题比对，返回差异描述列表（空 = 全部一致）。

    参数：
        records   {qid: gold 记录}（有 gold 的 16 题，含 "question"）
        expected  {qid: 卡8 重建出来的 top-k chunk_id 列表}

    要求：
      a) 对每道题调 retrieve_via_pipeline(pipeline, records[qid]["question"], mode)
      b) 与 expected[qid] 逐位比较。不一致就 append 一条描述，且【必须带上
         题号和两边序列】—— 报告要能直接看出差在第几位、差的是哪块，
         只写「不一致」等于让人再跑一遍
      c) 每题打一行进度（`[mode] id=16 ✅/❌`）：
         16 题 × 2 mode 会跑 32 次，出问题时要知道跑到哪了
      d) 这里【不要 raise】，把差异收集齐再交给 main 统一处理 ——
         与 gold._audit 同一个理由：一次跑完看到全部问题，
         比「修一个跑一次」快得多
    """
    resout = []
    for qid in records:
        chunk_ids = retrieve_via_pipeline(pipeline, records[qid]["question"], mode)
        exp = expected[qid]                      # 期望值按题号取，不是取字面量 "qid"
        if chunk_ids != exp:
            # 逐位找差异位置（1 开始计数）：报告要能直接看出差在第几位、差的是哪块
            n = max(len(exp), len(chunk_ids))
            bad = [i + 1 for i in range(n) if exp[i:i + 1] != chunk_ids[i:i + 1]]
            resout.append(f"id={qid} 差在第 {bad} 位 | expect={exp} | pipeline={chunk_ids}")
            print(f"[{mode}] id={qid} ❌")
        else:
            print(f"[{mode}] id={qid} ✅")
    return resout


# ------------------------------------------------------------
# 以下为已给：假 LLM / 复用卡8 的重建 / main
# ------------------------------------------------------------
class FakeLLM:
    """已给：让 pipeline.answer() 走完整链路但不真的调 LLM。

    rag.llm.ask 只用 llm.chat.completions.create(...) 一个入口，返回体只读
    choices[0].message.content 与 usage.prompt_tokens / completion_tokens ——
    所以三层嵌套的假对象就够，不需要真的 OpenAI 客户端（也就不会联网/花钱）。
    """

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kw) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[假回答：本卡不检生成段]"))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )


def load_rescue_module():
    """已给：把 scripts/11_rescue_analysis.py 当模块加载，复用它的 rebuild_top_k 与
    load_evidence（含 gold 指纹校验）。见文件头「为什么复用而不是抄一份」。"""
    path = ROOT / "scripts" / "11_rescue_analysis.py"
    spec = importlib.util.spec_from_file_location("rescue_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    """已给：装配 + 两 mode 汇总 + 不通过就 raise。"""
    rescue = load_rescue_module()
    ev, records, by_id = rescue.load_evidence()

    print(f"证据：{EVIDENCE_JSON.name}（生成于 {ev['generated_at']}）")
    print(f"服务默认 RETRIEVAL_MODE={RETRIEVAL_MODE}（本卡逐题显式传 mode，不受该默认值影响）")
    print(f"后处理参数：dedupe=True top_k={RERANK_TOP_K} quota_cap={SOURCE_CAP}\n")

    ranker = BGEReranker()
    pipeline = RAGPipeline(get_collection(), llm=FakeLLM(), ranker=ranker)

    failed: list[str] = []
    for mode in MODES:
        # 期望值 = 卡8 从卡7 落盘证据重建的 top-k（零 GPU，同一份 select_diverse）
        expected = {qid: rescue.rebuild_top_k(by_id[mode][qid]["hits"], RERANK_TOP_K, SOURCE_CAP)
                    for qid in records}
        diffs = verify_mode(pipeline, records, expected, mode)
        n_ok = len(records) - len(diffs)
        flag = "✅" if not diffs else "❌"
        print(f"\n[{mode}] 与卡7/卡8 口径逐位一致：{n_ok} / {len(records)}  {flag}")
        for d in diffs:
            print(f"  - {d}")
        failed += diffs

    print("\n=== 结论 ===")
    if failed:
        raise AssertionError(
            f"pipeline 接入后与卡7/卡8 口径不一致，共 {len(failed)} 处 —— "
            f"先查 mode 是否真的透传到了 _retrieve（这是最常见的接线错）")
    print(f"两个 mode × {len(records)} 题全部逐位一致 → 服务路径与实验脚本共用同一套检索口径。")
    print("同时确认：卡8 的零 GPU 重建逻辑与新鲜召回结果一致（三方交叉验证成立）。")


if __name__ == "__main__":
    main()