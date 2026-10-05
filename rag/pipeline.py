"""编排层 | Day9：把「召回 → 精排 → 去重 → 截断 → 生成」串成一次调用。

为什么要有 pipeline（今天重构的核心产出）：
  Day10 要埋 Langfuse trace，Day11 要包成 FastAPI，Day12 要改成 SSE 流式——
  它们都只需要「调一次 answer() 拿结构化结果」，不必知道内部用了几个模型、几步检索。
  这就是「最小可运行 → 抽象公共接口」的收益：调用方与实现细节解耦。

两个入口（Day12 起）：
  answer()          一次性返回完整结果
  answer_stream()   检索完成后逐 token 吐出（SSE 用），检索段与 answer() 共用 _retrieve()

硬边界：单轮问答。
"""
import time

from .config import LLM_MODEL, RECALL_K, RERANK_TOP_K, RETRIEVAL_MODE, resolve_prompt
from .fusion import hybrid_retrieve
from .llm import ask, ask_stream, format_sources, get_llm
from .retriever import BGEReranker, dedupe_adjacent, select_diverse
from .trace import nullspan


class RAGPipeline:
    def __init__(self, collection, llm=None, ranker=None):
        # 依赖注入——collection / llm / ranker 都可以由外部传进来复用。
        # 评测时三个组共用一个 ranker，避免重复加载模型到显存；测试时也能塞假对象。）
        self.collection = collection
        self.llm = llm or get_llm()
        self.ranker = ranker or BGEReranker()

    # ------------------------------------------------------------
    # 检索段：answer / answer_stream 共用
    # ------------------------------------------------------------
    def _retrieve(self, question: str, dedupe: bool,
                  top_k: int | None, quota_cap: int | None, span,
                  mode: str | None = None) -> tuple:
        """召回 → 精排 → 去重 → 配额截断，返回 (hits, recall, ranked, recall_ms, rerank_ms, k)。

        为什么 Day12 要把这段从 answer() 里抽出来：
            流式只改【生成】那一段，检索完全相同。若把检索抄一份进 answer_stream，
            两个入口就有了两条独立演进的检索路径 —— Day14 的最终指标两边都要读，
            口径必须只有一份。与 Day9 抽 format_context 是同一个理由。

        顺序不能错：必须「精排 → 去重 → 截断」。
            先截断到 top3 再去重，去重只会让槽位空掉（3 条变 2 条）；
            先去重再截断，被去重腾出的槽位才能让「另一个来源」的块补进来。
            配额的选取与截断天然是一步（select_diverse 直接返回 k 条），所以它在去重之后。

        mode（W1 卡9）：None → 读 config.RETRIEVAL_MODE；"dense" / "hybrid"。
            ⚠️ 这里【故意不写 if mode 分支】：mode="dense" 时 hybrid_retrieve 内部
               走的就是 retrieve，于是服务路径与卡7 的实验脚本共用【除召回段以外的
               全部代码】。一旦这里分叉成两条调用链，卡7 的 A/B 结论就不再适用于
               服务路径 —— 那个 A/B 的前提正是「两组只有召回段不同」。
        """
        mode = mode or RETRIEVAL_MODE
        with span("retrieval"):                       # 父 span：整段检索
            t0 = time.perf_counter()
            with span("recall"):                      # 子 span：召回（含 query 编码）
                recall = hybrid_retrieve(self.collection, question, RECALL_K, mode=mode)
            t1 = time.perf_counter()
            with span("rerank+dedupe"):               # 子 span：精排 + 去重
                ranked = self.ranker.rerank(question, recall)   # 全量降序，此刻还不能截断
                if dedupe:
                    ranked = dedupe_adjacent(ranked)            # 去重腾出槽位
            t2 = time.perf_counter()

        # 截断放在去重之后：显式给了 top_k 就用它，否则统一用 RERANK_TOP_K。
        # 【Day14 修正】原写法在 top_k=None（= API/前端默认）时让 cross 题走 CROSS_TOP_K=5，
        # 而 config 里那行注释写着「Day9 已否：纯负收益，保留仅作记录」—— 一份被否掉的结论
        # 还活在默认路径上，副作用有三个：
        #   ① 「服务默认行为」与「交付指标口径（final 组 top_k=3）」在 cross 题上讲的是两件事；
        #   ② /health 报出的 default_top_k=3 变成假话；
        #   ③ Day14 的验收表会拿一个被否过的参数当"服务默认值"去解释指标。
        # 已否的结论必须在【生效路径】上真正失效，而不是只写在注释里。
        k = top_k if top_k is not None else RERANK_TOP_K
        if quota_cap:
            hits = select_diverse(ranked, k, quota_cap)   # 选取即截断，治同源垄断
        else:
            hits = ranked[:k]

        # perf_counter() 返回的是「时刻」（秒），相减才得到耗时；再 *1000 换成毫秒。
        # 返回 mode 而不是让调用方自己再算一遍：与 k 同理（k 也是取实际生效的值，
        # 不是调用方传进来的 None）—— 结果里要能看到【这一跑真正用了哪条召回路径】。
        return hits, recall, ranked, (t1 - t0) * 1000, (t2 - t1) * 1000, k, mode

    def answer(self, question: str, qtype: str = "in",
               dedupe: bool = True, top_k: int | None = None,
               quota_cap: int | None = None, prompt_version: str = "v1",
               mode: str | None = None, tracer=None) -> dict:
        """跑完整链路并返回结构化结果。

        参数：
            qtype      题类型（"in"/"cross"/"partial"/"out"）—— 只作 trace 元数据，不影响条数
            dedupe     是否做相邻块去重（A/B 实验的开关）
            top_k      显式指定送进 prompt 的条数；None = RERANK_TOP_K（=3，服务默认口径）
            quota_cap  同源配额：同一 source 最多占几条；None = 关闭（不配额，直接截断）
            prompt_version  system prompt 的版本（config.PROMPTS 的键，Day10 卡 4 的变量）
            mode       召回路径："dense" / "hybrid"；None = config.RETRIEVAL_MODE（默认 dense）。
                       W1 卡9 新增。默认不切 hybrid 的理由写在 config 那行注释里。
            tracer     rag.trace.Tracer 实例（Day10 卡 5）；None = 不埋点。
                       传了就返回 trace_id，并在 data/traces/traces.jsonl 追加一条记录。
                       为什么是可选参数而不是默认开：eval 的 90 次调用不该产 90 条 trace 噪声。

        提示：所有耗时都用 time.perf_counter()（单调时钟，不受系统时间调整影响）。
        """
        # 埋点：tracer 为空时 span 退化成 nullspan（空操作），业务逻辑一行都不用分叉
        tr = tracer.begin(question, qtype=qtype, prompt_version=prompt_version,
                          dedupe=dedupe, top_k=top_k, quota_cap=quota_cap,
                          mode=mode or RETRIEVAL_MODE) if tracer else None
        span = tr.span if tr else nullspan

        hits, recall, ranked, recall_ms, rerank_ms, k, mode = self._retrieve(
            question, dedupe, top_k, quota_cap, span, mode)
        # prompt 版本在这里解析成正文再往下传：快速失败——版本写错时直接报错，
        # 而不是把空 system prompt 发出去、拿一堆莫名其妙的答案回来做实验。
        system_prompt = resolve_prompt(prompt_version)

        with span("generation", model=LLM_MODEL):     # 生成 span：model 是 span 属性，token/成本在 finish 里补
            t2 = time.perf_counter()
            gen = ask(self.llm, question, hits, system_prompt)
            gen_ms = (time.perf_counter() - t2) * 1000

        out = {
            "answer": gen["answer"], "hits": hits, "sources": format_sources(hits),
            "prompt_version": prompt_version,           # 埋点时要知道这条回答出自哪版 prompt
            "recall_ms": recall_ms, "rerank_ms": rerank_ms,
            "retrieve_ms": recall_ms + rerank_ms,       # 检索 = 召回 + 精排/去重，不含生成
            "gen_ms": gen_ms,
            "prompt_tokens": gen["prompt_tokens"], "completion_tokens": gen["completion_tokens"],
            "cost": gen["cost"],
            "n_recall": len(recall), "n_dedup": len(ranked),   # n_dedup = 去重后剩下的条数
            "top_k": k,                                        # 实际生效的条数，不是传入的 None
            "retrieval_mode": mode,                            # 实际生效的召回路径（同上）
            "trace_id": tr.trace_id if tr else None,            # Day10 卡 5：未埋点时为 None
        }
        if tr:
            tr.finish(out, n_recall=len(recall), n_dedup=len(ranked))   # 落盘 traces.jsonl
        return out

    # ------------------------------------------------------------
    # 流式入口（Day12 卡 3）
    # ------------------------------------------------------------
    def answer_stream(self, question: str, qtype: str = "in",
                      dedupe: bool = True, top_k: int | None = None,
                      quota_cap: int | None = None, prompt_version: str = "v1",
                      mode: str | None = None, tracer=None):
        """流式版 answer()：检索照旧一次做完，生成改成边收边吐。

        yield 的事件（纯数据 dict，HTTP 层的 SSE 包装放在 api/main.py ——
        编排层不认识 event:/data:，那是传输协议的事）：

            {"type": "meta",    trace_id / prompt_version / top_k / retrieval_mode /
                                n_recall / n_dedup / recall_ms / rerank_ms / retrieve_ms}
            {"type": "sources", "hits": [...]}    检索一完成就发，前端可先渲染来源
            {"type": "delta",   "text": "..."}    转发 ask_stream 的增量
            {"type": "done",    answer / prompt_tokens / completion_tokens / cost /
                                gen_ms / trace_id}

        为什么把 sources 排在 delta 之前（而不是照计划放在最后）：
            检索只占 9% 的耗时（Day10 实测），正文流出前先把来源推过去，
            前端就能在 [1] 刚出现在文本里时立刻有出处可点。顺序定了就不能改 ——
            Day13 的前端是按事件顺序写的状态机。

        一个用普通函数写不出来的点：with span("generation") 跨越了多次 yield。
            生成器每次 yield 只是【挂起】而不是退出，所以这个 span 直到整条流结束
            才执行退出、才记时长 —— 于是它的时长天然等于「整段流的时长」，
            而不只是「第一个 token 的耗时」。这正是流式埋点想要的口径。

        注意：本函数是生成器 —— 函数体在第一次 next() 前一行都不会执行，
        所以【参数校验挪到了路由层】（见 config.resolve_prompt 的说明）。
        """
        tr = tracer.begin(question, qtype=qtype, prompt_version=prompt_version,
                          dedupe=dedupe, top_k=top_k, quota_cap=quota_cap,
                          mode=mode or RETRIEVAL_MODE) if tracer else None
        span = tr.span if tr else nullspan

        hits, recall, ranked, recall_ms, rerank_ms, k, mode = self._retrieve(
            question, dedupe, top_k, quota_cap, span, mode)
        system_prompt = resolve_prompt(prompt_version)

        yield {
            "type": "meta", "trace_id": tr.trace_id if tr else None,
            "prompt_version": prompt_version, "top_k": k,
            "retrieval_mode": mode,
            "n_recall": len(recall), "n_dedup": len(ranked),
            "recall_ms": recall_ms, "rerank_ms": rerank_ms,
            "retrieve_ms": recall_ms + rerank_ms,
        }
        yield {"type": "sources", "hits": hits}

        with span("generation", model=LLM_MODEL):
            t2 = time.perf_counter()
            gen = None
            for ev in ask_stream(self.llm, question, hits, system_prompt):
                if ev["type"] == "delta":
                    yield ev                      # 立刻往下传，不在本层攒
                else:
                    gen = ev                      # done 事件先扣下，等 span 收尾后再发
            gen_ms = (time.perf_counter() - t2) * 1000

        out = {
            "answer": gen["answer"], "hits": hits, "sources": format_sources(hits),
            "prompt_version": prompt_version,
            "recall_ms": recall_ms, "rerank_ms": rerank_ms,
            "retrieve_ms": recall_ms + rerank_ms, "gen_ms": gen_ms,
            "prompt_tokens": gen["prompt_tokens"], "completion_tokens": gen["completion_tokens"],
            "cost": gen["cost"],
            "n_recall": len(recall), "n_dedup": len(ranked), "top_k": k,
            "trace_id": tr.trace_id if tr else None,
        }
        if tr:
            tr.finish(out, n_recall=len(recall), n_dedup=len(ranked))

        yield {
            "type": "done", "answer": out["answer"], "trace_id": out["trace_id"],
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "cost": out["cost"], "gen_ms": gen_ms,
        }
