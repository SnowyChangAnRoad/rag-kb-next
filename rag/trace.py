"""可观测层 | Day10 卡 5：本地自研 trace（零账号方案，用户已拍板）。

是什么：
    RAG运行的记录仪，记录了问答的完整信息(调用链、检索明细、生成用量、延迟和反馈)
主要目的：
    记录RAG运行时的结构化数据，用于观测以及后续复盘优化

为什么自研而不是接 Langfuse 云账号：
    本阶段要证明的是「我懂 trace / span / generation / score 这套模型」，不是「我会点一个 SaaS 后台」。
    自研版把 Langfuse 的四个概念原样落成 jsonl：零账号、零网络依赖、可 git diff、可 grep ——
    而且 Day11 包 FastAPI、Day12 改 SSE 时，同一套 span 直接复用，不用再引一层 SDK。

数据形态（append-only，两个文件，见 config）：
    data/traces/traces.jsonl   一次问答一行 JSON（trace + 它的 spans + 检索明细 + 生成用量）
    data/traces/scores.jsonl   一条反馈一行 JSON，靠 trace_id 关联到某条 trace
    —— 为什么反馈不写回 traces.jsonl：trace 是「当时发生了什么」的事实，反馈是「事后补的评价」，
       写入时间不同、写入方也不同（Day13 前端点按钮时问答早就结束了）。分开存 → 补录反馈
       永远不会破坏历史 trace；查询时用 trace_id 做 join 即可。

概念对齐（通关会问）：
    trace       一次问答的完整记录，trace_id 串起所有 span 与 score
    span        一次操作的起止时间戳 + 耗时，可嵌套 → 这就是「检索 → 生成」层级的来源
    generation  特殊的 span：额外挂 model / prompt_tokens / completion_tokens / cost
    score       挂在 trace 上的评价（这里只有 👍/👎）

用法（在项目根 rag-kb 下执行）：
    python -m rag.trace selftest              零 API 自检：造一条假 trace 走完整链路
    python -m rag.trace list                  最近 10 条 trace 一览
    python -m rag.trace show <trace_id>       层级树 + 延迟分解 + token/成本
    python -m rag.trace stats                 全量汇总（含 👍/👎 计数）
    python -m rag.trace feedback <id> up|down "备注"     补录一条反馈
"""

import json
import time
import uuid

from contextlib import contextmanager

from .config import LLM_MODEL, SCORES_JSONL, TRACES_JSONL

# 👍=1.0 / 👎=0.0：Langfuse 的 score 是数值型（可加权、可求均值），value 只是给人看的标签
FEEDBACK_SCORE = {"up": 1.0, "down": 0.0}


def new_trace_id() -> str:
    """形如 20260922-104233-8f3a。时间前缀方便人肉排序，短后缀防同秒碰撞。"""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


# append-only 落盘，一行一条 JSON。
def _append(path, record: dict) -> None:
    # 不缩进是有意的 —— jsonl 的每一行必须独立可解析，缩进只会让文件变大且无法逐行读。
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# 加载全部 trace（append-only 落盘，一行一条 JSON。）
def load_traces() -> list[dict]:
    if not TRACES_JSONL.exists():
        return []
    with open(TRACES_JSONL, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# 读全部反馈并按 trace_id 归组 —— 这就是「join」那一步
def load_scores() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not SCORES_JSONL.exists():
        return out
    with open(SCORES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rec["trace_id"], []).append(rec)
    return out


@contextmanager
def nullspan(name: str = "", **attrs):
    """未开埋点时的空操作 —— 让 pipeline 里不必写 if/else 分支。"""
    yield


# ------------------------------------------------------------
# 一条 trace
# ------------------------------------------------------------
class Trace:
    """一次问答的埋点容器。__init__ / finish 已给，span 留 TODO-1。"""

    def __init__(self, question: str, qtype: str = "in", prompt_version: str = "v1", **config):
        self.trace_id = new_trace_id()
        self.ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.question = question
        self.qtype = qtype
        self.prompt_version = prompt_version
        # 本次调用的开关（dedupe / top_k / quota_cap）—— 复盘「这条 trace 出自哪一组实验」全靠它
        self.config = config
        self.spans: list[dict] = []
        self.retrieval: dict = {}
        self.generation: dict = {}
        self.totals: dict = {}
        self._t0 = time.perf_counter()   # 相对时间轴的零点
        self._stack: list[str] = []      # 当前「打开着」的 span 名，用来推 parent

    @contextmanager
    def span(self, name: str, **attrs):
        """TODO-1：开一个 span 并自测耗时，退出时落进 self.spans。

        为什么用上下文管理器（with）而不是「手动 t = perf_counter() ... append(...)」：
            ① 嵌套关系由「进入/退出」天然表达 —— with 里再开一个 with 就是子 span，
               调用方不用自己传 parent，也就不会传错；
            ② 异常路径不用管：finally 里统一收尾，span 不会「漏关」而让后续 span 认错爹。

            **attrs 让调用方可以直接写 span("generation", model="deepseek-v4-flash")。
        """
        parent = self._stack[-1] if self._stack else None
        start = time.perf_counter()
        self._stack.append(name)         # ← 进入：把自己变成后面 span 的父级
        try: yield                       # ← 把控制权交回 with 块里的代码
        finally:
            end = time.perf_counter()
            self._stack.pop()            # ← 退出：必须与 append 严格配对
            self.spans.append({
                "name": name, "parent": parent,
                "start_ms": (start - self._t0) * 1000,   # 减零点 → 相对时间轴；再 *1000 → 毫秒
                "end_ms": (end - self._t0) * 1000,
                "duration_ms": (end - start) * 1000, **attrs})


    # 把 pipeline 的返回值整理成 trace 记录并落盘 traces.jsonl
    def finish(self, out: dict, n_recall: int, n_dedup: int) -> dict:
        # 只存 chunk 预览、不存全文 —— 一次 top3 的全文能让 jsonl 迅速不可读；
        # 复盘「证据有没有进 prompt」看 chunk_id + 预览就够了（要全文去 chunks.jsonl 查）。
        self.retrieval = {
            "n_recall": n_recall, "n_dedup": n_dedup, "top_k": out["top_k"],
            "docs": [
                {"slot": i, "chunk_id": h["chunk_id"], "source": h["source"],
                 "page": h.get("page"), "url": h.get("url"),
                 "distance": h.get("distance"), "rerank_score": h.get("rerank_score"),
                 "text_chars": len(h["text"]), "preview": h["text"][:60].replace("\n", " ")}
                for i, h in enumerate(out["hits"], 1)
            ],
        }
        self.generation = {
            "model": LLM_MODEL,
            "answer": out["answer"], "answer_chars": len(out["answer"]),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "cost": out["cost"],
        }
        self.totals = {
            "retrieve_ms": out["retrieve_ms"], "gen_ms": out["gen_ms"],
            "total_ms": out["retrieve_ms"] + out["gen_ms"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "cost": out["cost"],
        }
        rec = {
            "trace_id": self.trace_id, "ts": self.ts,
            "question": self.question, "qtype": self.qtype,
            "prompt_version": self.prompt_version, "config": self.config,
            "spans": self.spans, "retrieval": self.retrieval,
            "generation": self.generation, "totals": self.totals,
        }
        _append(TRACES_JSONL, rec)
        return rec

    # 给本 trace 打反馈（cli 交互里用），实现见下面的 add_score
    def score(self, value: str, note: str = "") -> dict:
        return add_score(self.trace_id, value, note)


# 埋点入口
class Tracer:
    # pipeline.answer(..., tracer=tracer) 时才产生 trace。
    # 不传 tracer 时 pipeline 完全不受影响（eval 的 90 次调用就不该产 90 条 trace 噪声）。

    def begin(self, question: str, qtype: str = "in",
              prompt_version: str = "v1", **config) -> Trace:
        return Trace(question, qtype=qtype, prompt_version=prompt_version, **config)


# ------------------------------------------------------------
# 反馈（score）
# ------------------------------------------------------------
def add_score(trace_id: str, value: str, note: str = "") -> dict:
    """TODO-2：把一条 👍/👎 反馈挂到指定 trace 上。

    为什么反馈要独立成文件、独立成函数：
        trace 是「当时发生了什么」的事实，反馈是「事后补的评价」—— 写入时间不同、
        写入方也不同。分开存 → 补录反馈永远不会破坏历史 trace；查询时用 trace_id join。
    """
    if value not in FEEDBACK_SCORE:
        raise ValueError(f"反馈只能是 up/down，收到 {value!r}")
    rec = {"trace_id": trace_id, "name": "user_feedback", "value": value,
           "score": FEEDBACK_SCORE[value], "note": note,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}   # 与 trace 的 ts 同格式（cmd_show 按 [11:] 截时间）
    _append(SCORES_JSONL, rec)
    return rec

# ------------------------------------------------------------
# 查询（已给）—— 最近 n 条 trace 一览（trace 侧 + 反馈侧 join 后的样子）
# ------------------------------------------------------------
def cmd_list(n: int = 10) -> None:
    traces = load_traces()
    if not traces:
        print(f"还没有 trace（{TRACES_JSONL} 不存在或为空）—— 先跑 python -m rag.cli \"你的问题\"")
        return
    scores = load_scores()
    print(f"共 {len(traces)} 条 trace，显示最近 {min(n, len(traces))} 条：\n")
    print(f"{'trace_id':<24}{'时间':<11}{'题型':<8}{'prompt':<8}{'检索ms':>8}{'生成ms':>8}"
          f"{'输出tok':>9}{'成本元':>9}  反馈")
    print("-" * 104)
    for r in list(reversed(traces))[:n]:
        t = r["totals"]
        fb = "".join("👍" if s["value"] == "up" else "👎"
                     for s in scores.get(r["trace_id"], [])) or "-"
        print(f"{r['trace_id']:<24}{r['ts'][11:]:<11}{r['qtype']:<8}{r['prompt_version']:<8}"
              f"{t['retrieve_ms']:>8.1f}{t['gen_ms']:>8.0f}{t['completion_tokens']:>9}"
              f"{t['cost']:>9.4f}  {fb}")


# 打印一条 trace 的层级树 + 检索明细 + 生成用量 + 延迟分解
def cmd_show(trace_id: str) -> None:
    rec = next((r for r in load_traces() if r["trace_id"] == trace_id), None)
    if not rec:
        print(f"找不到 trace {trace_id}；用 python -m rag.trace list 看现有 id")
        return
    scores = load_scores().get(trace_id, [])

    print(f"\ntrace {rec['trace_id']}   {rec['ts']}   题型 {rec['qtype']}   prompt {rec['prompt_version']}")
    print(f"config：{rec['config']}")
    print(f"问题：{rec['question']}")

    # 缩进由 parent 链推出 —— Day11/12 再加新 span 也不用改这里
    depth: dict[str, int] = {}
    for s in rec["spans"]:
        depth[s["name"]] = 0 if not s.get("parent") else depth.get(s["parent"], 0) + 1
    print("\nspans（缩进 = 嵌套层级）：")
    for s in rec["spans"]:
        extra = {k: v for k, v in s.items()
                 if k not in ("name", "parent", "start_ms", "end_ms", "duration_ms")}
        tail = f"   {extra}" if extra else ""
        print(f"  {'    ' * depth[s['name']]}└─ {s['name']:<15}"
              f"{s['start_ms']:>8.1f} → {s['end_ms']:>8.1f} ms（{s['duration_ms']:.1f} ms）{tail}")

    r, g, t = rec["retrieval"], rec["generation"], rec["totals"]
    print(f"\n检索：候选池 {r['n_recall']} → 去重后 {r['n_dedup']} → 进 prompt {r['top_k']} 条")
    for d in r["docs"]:
        where = f"第{d['page']}页" if d.get("page") is not None else (d.get("url") or "网页")
        print(f"  [资料{d['slot']}] {d['chunk_id']:<22} {d['source']} {where} "
              f"rerank={d['rerank_score']:.4f} d={d['distance']:.4f} {d['text_chars']}字")
        print(f"          {d['preview']}")

    print(f"\n生成：{g['model']}  输出 {g['answer_chars']} 字")
    print(f"  输入 {g['prompt_tokens']} tok / 输出 {g['completion_tokens']} tok   成本 ¥{g['cost']:.4f}")
    print("  回答前几行：")
    for line in g["answer"].splitlines()[:6]:
        print(f"    {line}")

    total = t["total_ms"] or 1.0
    print(f"\n延迟分解：检索 {t['retrieve_ms']:.0f}ms（{t['retrieve_ms'] / total * 100:.0f}%）"
          f" + 生成 {t['gen_ms']:.0f}ms（{t['gen_ms'] / total * 100:.0f}%）"
          f" = 合计 {t['total_ms']:.0f}ms")
    fb = "、".join(f"{s['value']}({s['ts'][11:]})" for s in scores) or "（还没有反馈）"
    print(f"反馈：{fb}")


# 全量汇总 —— 延迟、token、成本、反馈计数
def cmd_stats() -> None:
    traces = load_traces()
    if not traces:
        print("还没有 trace 数据")
        return
    scores = load_scores()
    n = len(traces)

    def total(section: str, key: str):
        return sum(r[section][key] for r in traces)

    up = sum(1 for sc in scores.values() for s in sc if s["value"] == "up")
    down = sum(1 for sc in scores.values() for s in sc if s["value"] == "down")
    print(f"\n共 {n} 条 trace；其中 {len(scores)} 条有反馈（👍{up} / 👎{down}）")
    print(f"平均检索 {total('totals', 'retrieve_ms') / n:.1f} ms；"
          f"平均生成 {total('totals', 'gen_ms') / n:.0f} ms")
    print(f"累计输入 {total('totals', 'prompt_tokens')} tok / 输出 "
          f"{total('totals', 'completion_tokens')} tok / 成本 ¥{total('totals', 'cost'):.4f}")
    by_type: dict[str, int] = {}
    for r in traces:
        by_type[r["qtype"]] = by_type.get(r["qtype"], 0) + 1
    print("按题型：" + "  ".join(f"{k} {v} 条" for k, v in sorted(by_type.items())))


def selftest() -> None:
    """已给：零 API 自检 —— 造一条假 trace 走「span → finish → 落盘 → show」全链路。

    为什么值得有它：TODO-1/TODO-2 的正确性不该用「跑一次真问答花几分钱」来验证，
    拿假数据把结构跑通，再上真链路。这也是卡 5 的验收第一步。
    """
    tr = Tracer().begin("自检：检索到的资料是怎么和问题一起交给模型的？",
                        qtype="partial", prompt_version="v2",
                        dedupe=True, top_k=3, quota_cap=2)
    with tr.span("retrieval"):
        with tr.span("recall"):
            time.sleep(0.01)
        with tr.span("rerank+dedupe"):
            time.sleep(0.02)
    with tr.span("generation", model="fake-model"):
        time.sleep(0.01)
    fake_out = {
        "answer": "自检回答：资料里说的是……[1][2]\n（这条 trace 由 selftest 生成，不是真实问答）",
        "hits": [
            {"chunk_id": "RAG知识_p016_c001", "source": "RAG知识.pdf", "page": 16, "url": None,
             "distance": 0.3123, "rerank_score": 0.9812, "text": "自检用假知识块，仅用于验证结构。" * 4},
            {"chunk_id": "Transformer_p003_c002", "source": "Transformer--Attention Is All You Need.html",
             "page": None, "url": "https://zhuanlan.zhihu.com/p/xxx",
             "distance": 0.4011, "rerank_score": 0.7421, "text": "自检用假知识块二号。" * 4},
        ],
        "top_k": 2, "prompt_tokens": 731, "completion_tokens": 128, "cost": 0.00099,
        "retrieve_ms": 30.0, "gen_ms": 10.0,
    }
    tr.finish(fake_out, n_recall=10, n_dedup=4)
    add_score(tr.trace_id, "up", "自检反馈")
    print(f"自检通过：已写入 {TRACES_JSONL} 与 {SCORES_JSONL}")
    cmd_show(tr.trace_id)


def main() -> None:
    import sys
    args = sys.argv[1:]
    cmd = args[0] if args else "list"
    if cmd == "list":
        cmd_list(int(args[1]) if len(args) > 1 else 10)
    elif cmd == "show":
        cmd_show(args[1]) if len(args) > 1 else print("用法：python -m rag.trace show <trace_id>")
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "feedback":
        if len(args) < 3:
            print('用法：python -m rag.trace feedback <trace_id> up|down "备注"')
        else:
            print(f"已记录：{add_score(args[1], args[2], args[3] if len(args) > 3 else '')}")
    elif cmd == "selftest":
        selftest()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
