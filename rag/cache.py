"""W4 卡3 | 语义缓存：答案级缓存（问题向量近邻 + 阈值命中）。

它解决的问题 + 与卡2 的接口：
    卡2 把「检索几次」交给模型，代价是每次往返都实付（预4：C 组 = 2.5–3.0× A）。
    语义缓存把「同一问题的产物」留在本地 —— 问题向量与历史问题的余弦 ≥ 阈值即命中，
    整条链路（gate / 检索 / 自评 / 改写 / 生成）一步不走，成本塌到一次 query 编码。
    调用方是 run_agentic(..., cache=<SemanticCache>)（D 组）：缓存查询在 gate **之前**、
    回写在生成**之后**，且只有 prompt_variant == "v1" 的产物入缓存。

命中改变了失败面的形状（报告必须单列，对应 SoK「记忆投毒」）：
    缓存是系统主动记住的上轮产物。某轮产物一旦已被污染，后续命中会【原样重放】它，
    而且绕过了检索层的阈值过滤与重排 —— 一次污染被放大成长期污染。

两层判据（匹配方式不同，必须分开写）：
    · namespace —— model_key / collection / variant / prompt_version / 检索参数指纹。
      精确相等，用于分桶与失效：任一项变了 ⇒ 旧记录整体不适用。
    · 问题向量 —— 余弦相似度 ≥ 阈值，在语义邻域内命中。这是「语义」缓存的全部新意，
      也是它新引入的失败模式（相似 ≠ 同意图 ⇒ 假命中）的来源。

口径声明（预5 冻结项）：本轮 D 组第二趟是【原题重复】⇒ 两次编码的输入逐字相同、
余弦 ≈ 1.0 ⇒ 本轮实验对【阈值取值】没有分辨率（任何阈值都不漏命中）。阈值取经验值，
标定需「近义不同意图」的问题集 —— 归 W5。

自检：python -m rag.cache selftest（假 encoder + 临时文件；不加载模型、不联网）
"""
import json
import time

from pathlib import Path

from .config import (DATA_DIR, DEFAULT_EMBED, DEFAULT_VARIANT, RECALL_K, RERANK_TOP_K,
                     RETRIEVAL_MODE, RRF_K, resolve_variant)
from .embedding import encode_query

# 独立文件：缓存是「可删、删掉只影响速度不影响结果」的可再生数据，不与 W1–W3 的真值源、
# 也不与 w4_agentic_traces.jsonl 混写（同 trace 落盘的隔离纪律）。
CACHE_JSONL = DATA_DIR / "w4_semantic_cache.jsonl"

# 阈值（经验起手值）：0.9 是「同题必不误判」与「拦住一部分近义」之间的粗放落点。
# ⚠️ 口径声明（卡3 设计要点 §12.8）：本轮 D 组第二趟是【原题重复】⇒ 两次编码的输入逐字
#    相同、余弦 ≈ 1.0 ⇒ 本轮实验对阈值取值【没有分辨率】（任何阈值都不漏命中）。阈值标定
#    需要「近义不同意图」的问题集，归 W5；假命中用构造例单列，落 cache_match.similarity 留证。
CACHE_THRESHOLD = 0.9


# ---------------- 进 namespace 指纹的检索参数（卡3 决策：8 项全进） ----------------
# 判据：任一发生变化 ⇒ 送回生成段的 hits 集合或次序会变 ⇒ 旧缓存必须失效。
#   运行时 5 项 = pipeline 的检索入参（mode / top_k / quota_cap / dedupe / min_score）；
#   常量 3 项 = 会改变召回池或次序的 config 值（RECALL_K / RERANK_TOP_K / RRF_K）。
# 这份清单是【唯一】的枚举处：别的模块不许再抄一份（第二个真相源 = 静默漏项）。
CACHE_FINGERPRINT_KEYS = ("mode", "top_k", "quota_cap", "dedupe", "min_score",
                          "RECALL_K", "RERANK_TOP_K", "RRF_K")


def build_retrieval_params(*, mode: str | None = None, top_k: int | None = None,
                           quota_cap: int | None = None, dedupe: bool = True,
                           min_score: float = 0.0) -> dict:
    """按 CACHE_FINGERPRINT_KEYS 组装指纹 dict：运行时 5 项由调用方传，常量 3 项从 config 读。

    两条口径：
        · 常量必须**从 config 读**、不许手抄数值 —— 手抄等于「config 改了、指纹没变 ⇒
          旧缓存照命中」的静默失效；
        · `top_k=None` 归一成**实际生效值** `RERANK_TOP_K`（pipeline 的 None 就是这个意思）。
          不归一的话，「传 None」与「传 3」会算出两个 namespace ⇒ 同配置却假漏命中。
          `quota_cap=None`（= 关闭配额）保持 None，与「传了值」区分开。
    """
    return {"mode": mode or RETRIEVAL_MODE,
            "top_k": top_k if top_k is not None else RERANK_TOP_K,
            "quota_cap": quota_cap, "dedupe": dedupe, "min_score": min_score,
            "RECALL_K": RECALL_K, "RERANK_TOP_K": RERANK_TOP_K, "RRF_K": RRF_K}


# ------------------------------------------------------------
# namespace：把身份标签拼成可精确比对的字符串（卡3 设计要点 §12.2）
# ------------------------------------------------------------
def build_namespace(model_key: str, collection: str, variant: str,
                    prompt_version: str, retrieval_params: dict) -> str:
    """组装 namespace。

    成分 = 五个身份维度；检索参数以 dict 传入，**哪些项进指纹由调用方决定**（成分只有一个
    真相源：调用处），本函数只负责规范化与拼接。

    契约：
        · 确定性：同一组输入调用两次 ⇒ 逐字相同；
        · 与 dict 键序无关：内容相同、键序不同的 retrieval_params ⇒ 结果相同；
        · 任一维度发生变化（含 retrieval_params 里【任一】项）⇒ 结果发生变化。

    为什么遍历调用方给的 dict、而不是硬编码一份参数名清单：硬编码清单 == 第二个真相源 ——
    调用方多传一项（例如以后的 mode）会被这里静默忽略，于是「改了检索参数、缓存照样命中」
    的静默失效就发生了（§12.2 明确要求显式枚举只存在于一处）。
    为什么用 json.dumps 而不是字符串相加：相加既无分隔符（"a"+"bc" 与 "ab"+"c" 同形 ⇒
    不同身份可能撞成同一个 namespace），又要求每个值都是 str（bool / int / None 会 TypeError）。
    """
    return json.dumps(
        {"model": model_key, "collection": collection, "variant": variant,
         "prompt": prompt_version,
         "params": sorted((str(k), str(v)) for k, v in retrieval_params.items())},
        ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------
# 缓存本体
# ------------------------------------------------------------
def _json_native(obj):
    """json.dumps 的 default：numpy 标量 / 数组 → Python 原生。

    为什么必须在这里兜底：真实编码器返回 numpy，任何一个漏网的标量都会让 json.dumps 抛
    TypeError —— 而这次抛在实验跑到第 4 趟时，整轮白跑。自检长期用纯 Python float，
    这条路从没被走到，是典型「自检绿、真跑炸」。
    未知类型【直接报错】而不是退化成 str：静默退化 = 数字落盘时变成字符串 "0.99"，
    下游统计读到字符串却看不出异常，比崩溃更难发现。
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()        # np.float32 → Python float；ndarray → list of float
    raise TypeError(f"缓存记录里有不可 JSON 序列化的类型：{type(obj).__name__}")


class SemanticCache:
    """一次实验绑定一组身份（身份在 __init__ 固定，设计草案第 5 条选 (a)）。

    参数：
        model_key / variant / collection  嵌入模型身份；collection 缺省由注册表派生
                                          （项目硬约束：一个 collection 只绑一种嵌入模型，
                                          所以 collection 已经隐含了 variant 档位）
        prompt_version                    生成段版本
        retrieval_params                  检索参数身份（成分由调用方给，见 build_namespace）
        threshold                         相似度阈值（见 CACHE_THRESHOLD）
        path                              落盘文件；缺省 CACHE_JSONL
        encode                            依赖注入的编码器：question -> 已 L2 归一化的向量。
                                          缺省 = embedding.encode_query（查询侧入口，含
                                          query_instruction，与 retriever.py 同一个入口）。
                                          **必须走查询侧** —— 缓存向量要与「检索时真正用的那个
                                          查询向量」同源，用文档侧编码会静默口径漂移。
                                          自检注入假 encoder ⇒ 不加载模型、不联网。
    """

    def __init__(self, model_key: str = DEFAULT_EMBED, variant: str = DEFAULT_VARIANT,
                 collection: str | None = None, prompt_version: str = "v1",
                 retrieval_params: dict | None = None, threshold: float | None = None,
                 path: Path | str | None = None, encode=None):
        self.model_key = model_key
        self.variant = variant
        self.collection = collection or resolve_variant(model_key, variant)["collection"]
        self.prompt_version = prompt_version
        self.retrieval_params = dict(retrieval_params or {})
        missing = [k for k in CACHE_FINGERPRINT_KEYS if k not in self.retrieval_params]
        if missing:
            raise ValueError(
                f"retrieval_params 缺 {missing} —— 漏一项就是「改了检索参数、缓存照样命中」的"
                f"静默失效；请用 cache.build_retrieval_params(...) 组装")
        if threshold is None:
            raise ValueError("SemanticCache: 未给 threshold —— 见 CACHE_THRESHOLD / 设计要点 §12.8")
        self.threshold = threshold
        self.path = Path(path) if path is not None else CACHE_JSONL
        self.encode = encode or (lambda q: encode_query(model_key, [q], variant)[0])
        self.namespace = build_namespace(model_key, self.collection, variant,
                                         prompt_version, self.retrieval_params)
        self._records = self._load()

    # ------------------------------------------------------------
    # 查：命中判据
    # ------------------------------------------------------------
    def get(self, question: str) -> dict | None:
        """编码问题 → 与已加载记录逐条算余弦 → 取最大 → 与阈值比较。

        输出：未命中 ⇒ None；命中 ⇒ 下面这个 dict（键名固定，调用方按此读）：
            {"answer": str, "sources": str, "fused_hits": list[dict],
             "cache_match": {"similarity": float, "matched_qid": str,
                             "encode_ms": float, "lookup_ms": float}}

        实现要点（对应卡3 设计要点 §12）：
            · self._records 已由 _load 按 namespace 过滤 ⇒ 这里【不再】重复过滤：
              过滤只有一个真相源，重复一层会让以后改 _load 的人误判是谁在拦；
            · 向量已 L2 归一化（真路径由 embedding._postprocess 保证）⇒ **内积即余弦**，
              不在这里再手写 normalize；
            · 用 None 而非 0 作初值：余弦可取负值（反向问题），用 0 当「没找到」会把
              「无候选」与「候选相似度为 0」混成一个状态；
            · 并列时用 qid 兜底 ⇒ 结果与落盘顺序无关（落盘顺序会随文件重排 / 换机器而变）。
        """
        if not self._records:
            return None   # 空缓存直接返回：省掉一次 GPU 编码（不是「顺手加」的检查）

        t0 = time.perf_counter()
        query_vec = self.encode(question)
        t1 = time.perf_counter()

        best_sim: float | None = None
        best_rec: dict | None = None
        for rec in self._records:
            if len(rec["vec"]) != len(query_vec):
                # 维度不同必须当场报错：zip 会【静默截断】到较短一方，于是算出一个小得
                # 离谱的相似度却什么也不提示（同族：dim 不同却同类库的静默失准）。
                raise ValueError(
                    f"缓存记录 {rec.get('qid')!r} 是 {len(rec['vec'])} 维，"
                    f"本次查询向量是 {len(query_vec)} 维 —— namespace 本该拦住这种混档，"
                    f"请检查落盘文件或身份参数")
            sim = sum(a * b for a, b in zip(query_vec, rec["vec"]))
            if best_sim is None or sim > best_sim or (
                    sim == best_sim and str(rec["qid"]) < str(best_rec["qid"])):
                best_sim, best_rec = sim, rec
        t2 = time.perf_counter()

        if best_sim < self.threshold:
            return None
        return {"answer": best_rec["answer"],
                "sources": best_rec["sources"],
                "fused_hits": best_rec["fused_hits"],
                # similarity 也转原生 float：best_sim 是 numpy.float32，不转的话落 trace 时
                # 会被 agentic.dump_trace 的 default=str 静默写成 "0.99"（字符串），
                # 卡5 读 cache_match 拿到的就不是数字
                "cache_match": {"similarity": float(best_sim), "matched_qid": best_rec["qid"],
                                "encode_ms": (t1 - t0) * 1000,
                                "lookup_ms": (t2 - t1) * 1000}}


    # ------------------------------------------------------------
    # 写（已给：管道代码）
    # ------------------------------------------------------------
    def put(self, question: str, *, qid: str = "", answer: str,
            sources: str = "", fused_hits: list) -> bool:
        """写入一条记录并 append 落盘；返回 True = 真的写了一条。

        契约：
            · 只在【未命中】路径被调用 —— 命中趟直接返回，不回写（否则缓存自我繁殖）；
            · 幂等：同 namespace 下同 question 已有记录 ⇒ 不写、返回 False；
            · 记录的 vec 由 self.encode(question) 生成 = 与查询侧同一个 encoder ⇒ 同源。
        """
        if any(r.get("question") == question for r in self._records):
            return False
        rec = {"namespace": self.namespace, "question": question, "qid": qid,
               # 显式 [float(x)]：真路径 encoder 返回 numpy 数组 ⇒ list(...) 得到的是
               # list[np.float32]，json 不认 numpy 标量。顺带把点积累加精度提到 float64。
               "vec": [float(x) for x in self.encode(question)],
               "answer": answer, "sources": sources, "fused_hits": fused_hits}
        self._records.append(rec)
        self._append(rec)
        return True

    def __len__(self) -> int:
        """当前 namespace 下已加载的记录数（自检的幂等断言读它）。"""
        return len(self._records)

    # ------------------------------------------------------------
    # 落盘（已给）
    # ------------------------------------------------------------
    def _load(self) -> list[dict]:
        """读落盘文件，只留下 namespace 相同的记录。

        坏行直接报错并带行号，【不】静默跳过 —— 「文件里有读不懂的行」是必须看见的状态，
        静默跳过的后果是「缓存看起来在读、其实少了一半」（与 W1 卡3 陈旧 token 同族）。
        """
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.path} 第 {lineno} 行不是合法 JSON：{e}") from e
                if rec.get("namespace") == self.namespace:
                    out.append(rec)
        return out

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            # default=_json_native 是写入边界的兜底：以后任何字段再漏 numpy，也不会
            # 让实验跑到一半整轮白跑（详见 _json_native 的 docstring）
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_native) + "\n")


# ------------------------------------------------------------
# 零网络自检
# ------------------------------------------------------------
class _FakeEncoder:
    """假编码器：查表返回预置单位向量；表里没有的返回一个固定向量。

    单位向量是构造出来的（构造时就算好，不在代码里做归一化）—— 与契约「encode 必须返回
    已归一化向量」对应：真路径的归一化在 embedding._postprocess，缓存侧不再做第二遍。
    """

    def __init__(self, table: dict):
        self.table = table

    def __call__(self, q: str) -> list:
        return list(self.table.get(q, [0.0, 0.0, 1.0]))


def _selftest() -> None:
    """python -m rag.cache selftest：临时文件 + 假 encoder，不加载模型、不联网。"""
    import tempfile

    import numpy as np   # ⑫ 要复现真路径的 numpy 向量（假 encoder 返回的是纯 Python float）

    def ok(cond, label):
        assert cond, f"❌ {label}"
        print(f"  ✓ {label}")

    # 0.90 只是自检用的显式阈值；真阈值取 CACHE_THRESHOLD
    TH = 0.90
    Q = "向量检索是什么"
    NEAR = "向量检索是什么？"                     # 措辞极近、意图不同（假命中构造例）
    OTHER = "今天午饭吃什么"                       # 无关问题
    ENC = _FakeEncoder({Q: [1.0, 0.0, 0.0],
                        NEAR: [0.9990, 0.0447, 0.0],   # 余弦 ≈ 0.999
                        OTHER: [0.0, 1.0, 0.0]})

    def new_cache(tmp, **kw):
        base = dict(model_key="bge-small-zh", variant="raw", prompt_version="v1",
                    retrieval_params=build_retrieval_params(), threshold=TH, encode=ENC)
        base.update(kw)
        return SemanticCache(path=tmp, **base)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cache.jsonl"

        # ① namespace 的三个契约
        ns = build_namespace("bge-small-zh", "day6_kb", "raw", "v1", {"a": 1, "b": 2})
        ok(ns == build_namespace("bge-small-zh", "day6_kb", "raw", "v1", {"b": 2, "a": 1}),
           "① namespace：与 dict 键序无关")
        ok(ns != build_namespace("bge-small-zh", "day6_kb", "raw", "v1", {"a": 1}),
           "① namespace：检索参数少一项 ⇒ 变化")
        ok(ns != build_namespace("bge-small-zh", "day6_kb", "raw", "v1-nohit", {"a": 1, "b": 2}),
           "① namespace：prompt_version 变化 ⇒ 变化")

        c = new_cache(p)
        ok(len(c) == 0, "① 空缓存长度为 0")

        # ② 空缓存 ⇒ 未命中
        ok(c.get(Q) is None, "② 空缓存：同题也不命中（没有记录可比较）")

        # ③ 写入 → 同题命中
        wrote = c.put(Q, qid="q1", answer="向量检索是把文本编码成向量后比距离",
                      sources="[1] RAG知识.pdf 第15页\n", fused_hits=[{"chunk_id": "c1"}])
        ok(wrote is True and len(c) == 1, "③ put 写入一条")
        hit = c.get(Q)
        ok(hit is not None and hit["cache_match"]["similarity"] >= 0.999,
           "③ 同题 ⇒ 命中且 similarity ≥ 0.999")
        ok(hit["answer"] == "向量检索是把文本编码成向量后比距离"
           and hit["fused_hits"][0]["chunk_id"] == "c1",
           "③ 命中返回 answer 与 fused_hits")
        ok(hit["cache_match"]["matched_qid"] == "q1", "③ 命中记下被命中的 qid")
        ok(hit["cache_match"]["encode_ms"] is not None and hit["cache_match"]["lookup_ms"] is not None,
           "③ 命中落 encode_ms / lookup_ms（预5 的延迟口径靠它）")

        # ④ 幂等
        ok(c.put(Q, qid="q1", answer="重复写", sources="", fused_hits=[]) is False and len(c) == 1,
           "④ 同题重复 put ⇒ 不写、长度不变")

        # ⑤ 无关问题 ⇒ 未命中
        ok(c.get(OTHER) is None, "⑤ 余弦≈0 的问题 ⇒ 未命中")

        # ⑥ 假命中构造例：措辞极近、意图不同 ⇒ 仍会命中（这就是 badcase 的证据）
        near_hit = c.get(NEAR)
        ok(near_hit is not None and near_hit["cache_match"]["matched_qid"] == "q1",
           "⑥ 措辞极近但意图不同 ⇒ 仍命中（假命中，相似度已落盘可留证）")

        # ⑦ 失效：身份变了 ⇒ 整库不适用
        other_ns = new_cache(p, prompt_version="v1-nohit")
        ok(other_ns.get(Q) is None, "⑦ prompt_version 变化 ⇒ 旧记录不命中（失效判据生效）")

        # ⑧ 落盘存活：新实例（同身份）读到旧记录 ⇒ 模拟服务重启
        reopened = new_cache(p)
        ok(len(reopened) == 1 and reopened.get(Q) is not None,
           "⑧ 重建实例 ⇒ 从落盘读回，仍能命中（重启存活）")

        # ⑨ 维度不一致必须当场报错（zip 会静默截断到较短一方 ⇒ 算出错误相似度却不提示）
        with tempfile.TemporaryDirectory() as d2:
            c2 = new_cache(Path(d2) / "cache.jsonl")
            c2.put(Q, qid="q1", answer="a", sources="", fused_hits=[{"chunk_id": "c1"}])
            c2._records[0]["vec"] = [1.0, 0.0, 0.0, 0.0]   # 手工混档：4 维 vs 3 维查询向量
            raised = False
            try:
                c2.get(Q)
            except ValueError:
                raised = True
            ok(raised, "⑨ 记录维度 ≠ 查询维度 ⇒ 当场报错（不静默截断）")

        # ⑩ 相似度并列 ⇒ 用 qid 兜底，结果与写入顺序无关
        tie_enc = _FakeEncoder({Q: [1.0, 0.0, 0.0], "向量检索是干啥的": [1.0, 0.0, 0.0]})
        with tempfile.TemporaryDirectory() as d3:
            c3 = new_cache(Path(d3) / "cache.jsonl", encode=tie_enc)
            c3.put(Q, qid="q_b", answer="B", sources="", fused_hits=[])            # 先写大 qid
            c3.put("向量检索是干啥的", qid="q_a", answer="A", sources="", fused_hits=[])
            ok(c3.get(Q)["cache_match"]["matched_qid"] == "q_a",
               "⑩ 相似度并列 ⇒ 用 qid 兜底（与写入顺序无关）")

        # ⑪ 指纹缺项 ⇒ 当场报错（漏一项就是「改了检索参数、缓存照样命中」的静默失效）
        raised = False
        try:
            new_cache(p, retrieval_params={"top_k": 3})
        except ValueError:
            raised = True
        ok(raised, "⑪ retrieval_params 缺项 ⇒ 构造时当场报错（不是静默漏项）")

        # ⑫ 真路径的 encoder 返回 numpy ⇒ 落盘必须是原生 float
        #    （既不许抛 TypeError，也不许被 default=str 退化成字符串 "0.99"）
        #    自检长期用纯 Python float ⇒ 这条路从没被走到，真跑时在第 4 趟写缓存抛
        #    TypeError: Object of type float32 is not JSON serializable，整轮白跑。
        with tempfile.TemporaryDirectory() as d4:
            p4 = Path(d4) / "cache.jsonl"

            def np_enc(q):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

            c4 = new_cache(p4, encode=np_enc)
            c4.put(Q, qid="q1", answer="a", sources="", fused_hits=[{"chunk_id": "c1"}])
            last = json.loads(p4.read_text(encoding="utf-8").splitlines()[-1])
            ok(all(isinstance(x, float) for x in last["vec"]),
               "⑫ numpy 向量 ⇒ 落盘为原生 float（不抛错、不退化成字符串）")
            ok(isinstance(c4.get(Q)["cache_match"]["similarity"], float),
               "⑫ similarity 是原生 float（落 trace 不会被写成字符串）")

    print("\n✅ cache selftest 全通过")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print(__doc__)
