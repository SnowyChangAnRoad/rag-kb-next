"""服务层 | Day11：把 RAGPipeline 包成 HTTP 服务（FastAPI + uvicorn）。

为什么要有这一层：
    Day9 的 pipeline.answer() 已经做到「一次调用拿结构化结果」；Day11 要做的是把它
    暴露成别的程序能调的接口 —— 前端（Day13）、curl、评测脚本都通过 HTTP 说话，
    不必再各自装 torch / chromadb / 模型权重。这就是「模块化 → 服务化」的收益。

接口清单：
    GET  /health        探活 + 知识库状态
    POST /chat          一次问答（召回 → 精排 → 去重 → 配额 → 生成），可埋点返回 trace_id
    POST /ingest        读 data/processed/chunks.jsonl 全量重建向量索引
    POST /feedback      给某条 trace 补一条 👍/👎（Day10 卡 5 的 score，Day13 前端按钮用）
    POST /chat/stream   Day12：同一链路的 SSE 流式版（meta → sources → delta… → done）
    POST /files         Day12：上传 PDF → 解析 → 切块 → 写 chunks.jsonl（真相源）→ 入库
    GET  /documents     Day13：按 source 聚合，返回「库里有哪些资料」（前端知识库标签页用）
    GET  /              Day13：web/ 静态前端（双标签页：知识库管理 / 问答）

启动（项目根目录 rag-kb 下，需关闭沙箱）：
    uvicorn api.main:app --port 8000
    浏览器打开 http://127.0.0.1:8000/docs 自测（Swagger UI 就是今天的验收工具）

四个硬约束（通关会问，写之前先想清楚）：
    ① 重资源只加载一次。collection / bge-reranker / LLM 客户端全部放进 lifespan，
       挂到 app.state 上。若写在模块顶层，`--reload` 每改一行代码就重新加载一遍
       （十几秒 + 显存里堆两份模型），而且测试时无法替换。
    ② 路由写 `def`，不要写 `async def`。pipeline.answer 是同步阻塞的（GPU 前向 + 网络
       IO），FastAPI 对 `def` 路由会自动丢到线程池；写成 `async def` 会占住事件循环，
       第二个请求开始全部排队 —— 不报错，只是并发能力悄悄变成 0。
    ③ 出参走 Pydantic 模型白名单。hits 里带着整段 text，直出既浪费带宽又暴露内部结构；
       response_model 只放行该给的字段，多出来的字段会被自动丢掉。
    ④ 【Day12 新增】SSE 的生成器必须是同步 `def`，且参数校验必须在建流【之前】完成。
       前者与 ② 同源：`async def` 生成器里跑同步的 `for ev in pipeline.answer_stream(...)`
       会独占事件循环（第二个请求开始排队，且不报错）。
       后者是流式特有的坑：生成器的函数体在第一次 next() 前一行都不执行，
       等它抛 ValueError 时 HTTP 200 与响应头早已发出 —— 客户端只会看到「半截流」。
       所以 config.resolve_prompt 被从 pipeline 提到 config 顶层，由路由先调、再建流。
"""
import json
import shutil
import time

from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import (
    CHUNK_SIZE, RERANK_TOP_K, RETRIEVAL_MODE, SOURCE_CAP, UPLOAD_DIR, UPLOAD_MAX_MB, WEB_DIR,
    resolve_prompt,                     # TODO-9 用：路由层在建流前做参数校验
)
from rag.ingest import build_index, load_chunks, save_chunks
from rag.llm import hit_where, safe_preview
from rag.pipeline import RAGPipeline
from rag.retriever import get_client, get_collection
from rag.splitter import pdf_to_chunks      # TODO-11 用：与 scripts/05_split.py 同一份切块口径
from rag.trace import Tracer, add_score


# ------------------------------------------------------------
# 请求 / 响应模型（Pydantic）
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    """请求体 = 一次问答的全部旋钮。

    这些字段不是随便开的：它们与 Day9–10 的实验变量一一对应
    （top_k / quota_cap / prompt_version / dedupe），这样 Swagger 里就能复现任何一组实验。
    默认值取 Day10 结论最好的 C 组（quota_cap=2 + prompt v2，30 题正确率 100%）。
    """
    question: str = Field(..., min_length=1, description="用户问题")
    qtype: str = Field("in", description="题型 in/cross/partial/out，只写进 trace 供复盘/分组，不影响条数")
    top_k: int | None = Field(None, ge=1, le=10, description="送进 prompt 的资料条数；留空 = 用 RERANK_TOP_K（3）")
    quota_cap: int | None = Field(SOURCE_CAP, ge=1, le=5, description="同源配额：同一来源最多占几条")
    prompt_version: str = Field("v2", description="system prompt 版本：v1 或 v2")
    dedupe: bool = Field(True, description="是否做相邻块去重")
    trace: bool = Field(True, description="是否写一条 trace 到 data/traces/traces.jsonl")

class SourceItem(BaseModel):
    """来源条目 —— 注意这里【没有】text 全文字段，只给 preview。

    前端要展示的是「第几页、哪个文件、排第几」，不是几百字正文；
    需要正文时拿 chunk_id 去 chunks.jsonl 查（chunk_id 就是那把钥匙）。
    """
    slot: int                       # 与回答里的 [n] 对应，1 开始
    chunk_id: str
    source: str
    where: str                      # 「第13页」或网页 url
    distance: float | None
    rerank_score: float | None
    preview: str

class ChatResponse(BaseModel):
    """结构化回答 —— 字段名与 pipeline.answer 的返回保持一致（Day13 前端、Day14 指标都读它）。"""
    answer: str
    sources: list[SourceItem]
    prompt_version: str
    top_k: int
    n_recall: int
    n_dedup: int
    retrieve_ms: float
    gen_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    trace_id: str | None = None

class IngestResponse(BaseModel):
    collection: str
    chunks: int                     # 写完后库里的总块数
    elapsed_ms: float

class FeedbackRequest(BaseModel):
    trace_id: str
    value: str = Field(..., pattern="^(up|down)$", description="up=👍 / down=👎")
    note: str = ""

class UploadResponse(BaseModel):
    """上传结果 —— 必须能解释「我上传的页去哪了」，不能只回一个成功。

    pages / skipped 是刻意暴露的：上传 20 页却只入库 3 块时，
    接口要能说清另外 17 页是被判成目录页/残页剔了，还是扫描件根本没抽出文字
    （pdf_pages_to_docs 对空页直接跳过）。静默丢数据是上传接口最忌讳的行为 ——
    同 Day11 /health 不许把「索引坏了」伪装成 chunks:0。
    """
    filename: str
    pages: int                  # PDF 总页数（抽到文字的页）
    skipped: int                # 被过滤的页数（目录页 / 残页）
    chunks: int                 # 本次写入的块数
    total: int                  # 写完后库里的总块数
    elapsed_ms: float

class DocumentItem(BaseModel):
    """一份文档的聚合视图（Day13 知识库标签页的列表项）。

    为什么按 source 聚合而不是把 932 个块全发给前端：
        前端要回答的是「我库里有哪些资料」，不是「第几块写了什么」。
        聚合发生在服务端（一次 collection.get 拿 metadatas，纯内存分组），
        带宽从 932 条降到个位数条。
    """
    source: str                  # 文件名或网页标题，与 citation 里的来源一致
    chunks: int                  # 该来源的块数
    pages: int                   # 覆盖页数（网页块无 page，计 0）
    chars: int                   # 该来源总字数（元数据 char_count 之和）

class DocumentsResponse(BaseModel):
    collection: str
    n_documents: int
    documents: list[DocumentItem]


# ------------------------------------------------------------
# 生命周期：重资源只加载一次
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """TODO-1：服务启动时加载 collection 与 RAGPipeline，挂到 app.state 上；yield 之后做清理。

    提示（踩过的坑）：
        - RAGPipeline() 里会加载 bge-reranker-base 到 cuda，本机约十几秒 —— 这正是
          「不能放在模块顶层、也不能每次请求都建」的原因。
        - 不要用已废弃的 @app.on_event("startup")：lifespan 是它的替代品，
          且能保证「启动 → yield → 关闭」的顺序与异常清理。
        - 访问 ChromaDB 必须关闭沙箱，否则 chroma.sqlite3 报 disk I/O error（Day9 教训）。
    """
    app.state.collection = get_collection(get_client())
    app.state.pipeline = RAGPipeline(app.state.collection)
    print(f"已加载 {app.state.collection.count()} 条知识块({app.state.collection.name})")
    yield


app = FastAPI(
    title="RAG 知识库 API",
    description="Day11：把 rag.pipeline 包成 HTTP 服务。链路＝向量召回 → bge-reranker 精排 → 相邻块去重 → 同源配额 → DeepSeek 生成。",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS：浏览器同源策略默认禁止 http://localhost:5500 的页面调 127.0.0.1:8000 的接口。
# Day13 的前端就是这种情况，所以现在就放开 —— 否则那天会先卡在跨域上，与 RAG 无关。
# 注意 allow_origins=["*"] 不能和 allow_credentials=True 同时用（浏览器会直接拒绝携带凭证），
# 本项目前端不带 cookie，所以留 False。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# 路由
# ------------------------------------------------------------
@app.get("/health")
def health(request: Request) -> dict:
    """TODO-2：探活接口。返回服务与知识库状态，形如：
        {"status": "ok", "collection": "day6_kb", "chunks": 932, "default_top_k": 3}

    为什么要它：Day12 起前端要判断「后端起来没」、Day14 联调要能一眼看出是服务挂了
    还是库空了。chunks=0 就是「库空了」的显式信号，比 500 报错好排查。

    提示：`collection.count()` 会真的去查 Chroma（有 IO 开销），所以探活别做得太重 ——
    这里查一次是合理的；真正的「就绪探针」可以只返回 status。

    【批改：原来的 try/except Exception 已删除，原因见下】
    原写法把「Chroma 真坏了」和「服务还没就绪」合并成同一个 status:"bad"，并统一报
    chunks:0 —— 而 0 是编造出来的假数据。Day9 就真实踩过 HNSW 索引损坏，若当时挂着这个
    接口，你会看到 chunks:0，第一反应是「库空了」，排错方向直接跑偏（真实原因是索引文件坏了）。
    探活接口的职责是【暴露】故障，不是掩盖故障。所以：
      - 服务没就绪 → 503，语义是「暂时不可用，稍后重试」（这是可预期的中间态）
      - collection 真抛异常 → 不接，让它变成 500 + 堆栈，一眼定位
    另外改用 `request.app.state`（依赖注入）而不是模块级 `app`：测试时可以塞一个假 app
    进来，也就不用 import 本模块的全局对象。
    """
    collection = getattr(request.app.state, "collection", None)
    if collection is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪：collection 未加载")
    return {
        "status": "ok", "collection": collection.name,
        "chunks": collection.count(), "default_top_k": RERANK_TOP_K,
        # W1 卡9：召回路径现在是生效路径上的真变量，就该被报出来。
        # 直接读 config：这是【服务默认值】，不是某次请求实际用的值（那在 /chat 返回里）。
        "retrieval_mode": RETRIEVAL_MODE}


def to_sources(hits: list[dict]) -> list[SourceItem]:
    """TODO-4：把 pipeline 返回的 hits 转成 SourceItem 列表（出参白名单）。

    为什么值得单开一个函数：Day12 的 SSE、Day13 的前端、Day14 的指标都要同一份来源结构，
    写两遍必然漂移 —— 这与 Day9 抽 format_context 是同一个理由。
    """
    return [
        SourceItem(
            slot=i,                                  # 从 1 开始，与回答里的 [n] 对齐
            chunk_id=h["chunk_id"],
            source=h["source"],
            where=hit_where(h),
            distance=h.get("distance"),
            rerank_score=h.get("rerank_score"),
            preview=safe_preview(h["text"], 80),
        )
        for i, h in enumerate(hits, 1)
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """TODO-3：核心接口。一次问答，返回结构化结果。

    要求：
        - 从 `request.app.state` 取 pipeline（不要在函数里新建 RAGPipeline —— 那会重新加载模型）
        - 按 req.trace 决定是否埋点：`tracer = Tracer() if req.trace else None`，
          调 `pipeline.answer(req.question, qtype=req.qtype, dedupe=req.dedupe,
          top_k=req.top_k, quota_cap=req.quota_cap, prompt_version=req.prompt_version, tracer=tracer)`
        - 把 out 组装成 ChatResponse 返回（sources 走上面的 to_sources）
        - prompt_version 非法时 pipeline 会抛 ValueError → 转成 HTTPException(400)，
          让 Swagger 看到「是你参数写错了」，而不是 500 内部错误

    本函数签名是 `def` 而不是 `async def` —— 这是今天最重要的一行决定，通关会问。

    【批改：原实现 4 处必崩 + 2 处静默错数据，已修】
      ① `req.tracee` → AttributeError（拼写多一个 e），应为 req.trace。
      ② `answer["source"]` → KeyError：pipeline 返回的键是 sources（复数），
         而且它是 format_sources() 的【字符串列表】，与 SourceItem 结构不是一回事 ——
         正确做法是拿 answer["hits"] 过一遍 to_sources()。原代码把 TODO-4 写了却没调用。
      ③ `gen_ms=answer["retrieve_ms"]` → 静默错数据：生成耗时被填成检索耗时。
         Day10 实测检索占 9%、生成占 91%，前端看到的耗时构成会完全反掉。应为 out["gen_ms"]。
      ④ `top_k=RERANK_TOP_K` → 静默错数据：写死常量 3，而 pipeline 返回的 top_k 是
         【本次实际生效】的条数（Day14 起各题型统一 3，但显式传 top_k 时就不是 3）。
         写死会让 API 报出的实验参数失真。
      ⑤ 用 request.app.state 取 pipeline（模块级 app 也能跑，但注入版可测，且与 /health 统一）。
      ⑥ 补上 ValueError → 400：注意 try 只包住 pipeline.answer 这一次调用，
         而不是包住整个函数体 —— 包太宽会把真正的代码 bug 也伪装成「你的参数写错了」。
      局部变量改名为 out：避免 `answer["answer"]` 这种自我指涉，也避免和响应字段 answer 混淆。
    """
    pipeline = request.app.state.pipeline
    tracer = Tracer() if req.trace else None
    try:
        out = pipeline.answer(
            req.question, qtype=req.qtype, dedupe=req.dedupe, top_k=req.top_k,
            quota_cap=req.quota_cap, prompt_version=req.prompt_version, tracer=tracer,
        )
    except ValueError as e:              # prompt_version 非法 = 入参问题，不是服务故障
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ChatResponse(
        answer=out["answer"],
        sources=to_sources(out["hits"]),
        prompt_version=out["prompt_version"],
        top_k=out["top_k"],                      # 实际生效条数，不是请求里的 None
        n_recall=out["n_recall"],
        n_dedup=out["n_dedup"],
        retrieve_ms=out["retrieve_ms"],
        gen_ms=out["gen_ms"],
        prompt_tokens=out["prompt_tokens"],
        completion_tokens=out["completion_tokens"],
        cost=out["cost"],
        trace_id=out["trace_id"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: Request) -> IngestResponse:
    """TODO-5：重建向量索引。

    要求：读 chunks.jsonl → 全量 upsert 进 day6_kb → 返回写完后库里的总块数 + 耗时。

    语义说明（今天定的口径，Day12 再改造成「上传 PDF → 解析 → 切块 → 入库」）：
        所谓「重建」= 拿 chunks.jsonl 这份真相源全量 upsert 覆盖。
        不需要先删集合：chunk_id 相同即覆盖（build_index 内部用 upsert 而非 add）。
        Day9 那次「删集合重灌」是因为 HNSW 索引文件损坏，不是常规流程 —— 常规重建用不上。

    两个要想清楚的点（通关会问）：
        ① 这个接口是长任务（932 块要过 BGE 编码），几秒到几十秒。
           放在请求里等，客户端会超时；真实生产要改成后台任务 + 任务 id 轮询。
           今天先用同步版本，但要能说清它的边界在哪。
        ② 它不该被并发调用（两个请求同时 upsert 同一集合，语义混乱）。
           今天不实现锁，但要能说出「为什么这里不加锁是危险的」。

      这四处本质是同一个问题：**现成的 load_chunks / build_index 就在手边（骨架里已经 import 了），
      却重新手写了一遍读文件 + 写库的逻辑。** 重写一次就重踩一次 Day2/Day6 已经踩平的坑
      （id 重复、None metadata、分批、幂等）—— 能复用就复用，不只是少写代码，是少踩坑。

      保留你写对的两点：用 time.perf_counter()（单调时钟，比 time.time() 更适合测耗时）；
      chunks 取 collection.count() 而不是 len(chunks)（报的是库里的真实状态，不是意图）。
    """
    t0 = time.perf_counter()
    chunks = load_chunks()                                  # 真相源：data/processed/chunks.jsonl
    collection = request.app.state.collection
    total = build_index(collection, chunks)                 # 内部分批 upsert，返回写完后总数
    return IngestResponse(
        collection=collection.name,
        chunks=total,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """TODO-6：补一条 👍/👎 到 data/traces/scores.jsonl（Day10 卡 5 的 score）。

    要求：调 `add_score(req.trace_id, req.value, req.note)` 并返回落盘的那条记录。

    要想清楚的一个边界：trace_id 不存在时 add_score 不会报错，会写出一条「孤儿反馈」
    （查询时 join 不到任何 trace）。要不要先校验 trace_id 存在？两种做法的取舍是什么？
    """
    rec = add_score(req.trace_id, req.value, req.note)
    return rec


# ============================================================
# Day12 卡 2：SSE 流式问答
# ============================================================
def sse_frame(event: str, data: dict) -> str:
    """TODO-7：把一个事件拼成一帧 SSE 文本（协议见 MDN「使用 Server-Sent Events」）。

    SSE 的帧格式＝「若干字段行 + 一个空行」：
        event: delta
        data: {"text": "向量"}
        <这里必须有一个空行>

    最后那个空行是【帧结束标记】。少写它不会报错，客户端会一直等下一帧 ——现象是「curl 里什么都没有 / 前端一个字都不出」，属于最难查的那类错。
    """
    return f"event: {event}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n"


def sse_events(req: ChatRequest, pipeline) -> Iterator[str]:
    """TODO-8：把 pipeline.answer_stream 的 dict 事件翻译成 SSE 帧。

    answer_stream 产出四种事件（详见 rag/pipeline.py 文档字符串）：
        meta     链路参数：trace_id / prompt_version / top_k / n_recall / n_dedup / 各段耗时
        sources  检索结果 hits —— 注意它排在 delta 【之前】，前端才能先有出处可点
        delta    增量正文
        done     完整答案 + prompt_tokens / completion_tokens / cost / gen_ms

    要求：
        ① 逐个转发，`for ev in pipeline.answer_stream(...)`，本层不攒内容
           （攒了就等于把流式退化成一次性返回，白做）
        ② sources 必须先过 to_sources(ev["hits"]) 再序列化：hits 里带整段 text，
           直推既浪费带宽又暴露内部结构（Day11 硬约束③）。
           SourceItem 是 Pydantic 模型 → 用 .model_dump() 转成可 json 序列化的 dict。
        ③ delta 只带 {"text": ...}，别把整个事件 dict 塞进去
        ④ 本函数是【同步生成器】：`def` + `yield`（硬约束④）
        ⑤ 埋点：`tracer = Tracer() if req.trace else None`，传给 answer_stream

    一个想清楚再加分的点：客户端中途断开（浏览器关标签页）时，本生成器会被 close()，
    span 与 tr.finish 都不会执行 → 这条 trace 不落盘。要不要 try/finally 兜住，
    是个真实取舍（今天不实现，写进通关问题）。
    """
    tracer = Tracer() if req.trace else None
    for ev in pipeline.answer_stream(
            req.question, qtype=req.qtype, dedupe=req.dedupe, top_k=req.top_k,
            quota_cap=req.quota_cap, prompt_version=req.prompt_version, tracer=tracer,
    ):
        etype = ev["type"]

        if etype == "sources":
            # sources：必须先过 to_sources 做白名单，再 model_dump 成 JSON 可序列化 dict。
            # 原始 hits 里带整段 text，直推既浪费带宽又暴露内部结构。
            sources = [s.model_dump() for s in to_sources(ev["hits"])]
            yield sse_frame("sources", {"sources": sources})
        elif etype == "delta":
            # delta：只带增量正文，不要把整个 ev 塞进去
            yield sse_frame("delta", {"text": ev["text"]})
        elif etype == "done":
            # done：完整答案 + token / cost / gen_ms 等
            yield sse_frame("done", {k: v for k, v in ev.items() if k != "type"})
        else:
            # 未知事件兜底：原样转发，但去掉 type，避免和 SSE event 名重复
            yield sse_frame(etype, {k: v for k, v in ev.items() if k != "type"})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """TODO-9：SSE 路由。三步，顺序不能错：

        ① 【建流之前】校验参数：`resolve_prompt(req.prompt_version)`
           非法版本要在这里抛 HTTPException(status_code=400, detail=...)。
           为什么不能留给 pipeline 抛：answer_stream 是生成器，函数体在第一次 next()
           之前一行都不执行；等它抛 ValueError 时 HTTP 200 与响应头早已发出，
           客户端只会看到「半截流」，永远变不成 400。
        ② 取 pipeline：`request.app.state.pipeline`（新建 RAGPipeline 会重新加载模型）
        ③ 返回 `StreamingResponse(sse_events(req, pipeline),
             media_type="text/event-stream", headers={...})`

    headers 两个，各治一种病（都写）：
        Cache-Control: no-cache        别让中间层缓存住这条流
        X-Accel-Buffering: no          关掉 Nginx 类反代的响应缓冲 ——
                                       缓冲会把「逐 token」攒成「最后一次性给你」

    为什么本函数是 `def` 不是 `async def`：StreamingResponse 拿到同步迭代器时会用
    iterate_in_threadpool 逐块去取，取块的动作（含 pipeline 的同步生成器）落在
    线程池里，事件循环不被占住 —— 与硬约束②同一条原理。
    """
    # ① 建流之前先校验参数。
    # resolve_prompt 非法版本会抛 ValueError，这里立刻转成 400。
    # 不能留给 pipeline.answer_stream 抛：生成器函数体在第一次 next() 前一行都不执行，
    # 等它抛 ValueError 时 HTTP 200 和响应头早已发出，客户端只能看到半截流。
    try:
        resolve_prompt(req.prompt_version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # ② 取全局单例 pipeline，绝不能在这里新建 RAGPipeline()
    pipeline = request.app.state.pipeline

    # ③ 返回 SSE 流。
    # sse_events 是同步生成器，StreamingResponse 会用 iterate_in_threadpool 逐块取，
    # 不会占住事件循环。
    return StreamingResponse(
        sse_events(req, pipeline),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Day12 卡 3：上传 PDF → 解析 → 切块 → 入库
# ============================================================
@app.post("/files", response_model=UploadResponse)
def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """TODO-10 / TODO-11：上传 PDF 并入库。按下面 ①–⑤ 实现。

    为什么本路由用 `def`（Day11 那个判断在这里的答案是「照样 def」）：
        本路由全程是同步阻塞活（落盘 + pdfplumber 解析 + BGE 编码）。
        写成 `async def` 然后在里面 `await file.read()`，只有读文件那一步变异步，
        紧接着的解析/编码仍然同步卡住事件循环 —— 得不偿失。
        真要 `async def`，就得把重活 `await asyncio.to_thread(...)` 挪走（今天不做）。

    六步：校验 -> 落盘到 UPLOAD_DIR -> 解析切块 -> 写真相源 -> 入库 -> 组装

    【Day13 验收后补的修复】第 ④ 步「写 chunks.jsonl」是后加的：
        原来上传只写向量库。而 Day9 定的口径是「chunks.jsonl 是真相源、向量库可重建」——
        上传入口绕过真相源，等于在库里造了一批「重建追不回」的块：
        Day13 验收上传的 Redis全套学习笔记.pdf 401 块就只在库里，一旦 POST /ingest
        重建（或 HNSW 损坏后按老办法重建），这 401 块会静默消失、库从 932 掉回 531。
        所以上传必须同时落真相源，且顺序是【jsonl 先、库后】（失败方向更容易收拾）。

    两个边界（通关会问，今天不实现，要能说清）：
        - 同名文件重复上传：doc_id 用文件 stem → chunk_id 相同 → upsert 覆盖，天然幂等。
          但【内容变短时会残留旧块】：旧版 12 块、新版 8 块，第 9–12 块还在库里，
          提问时仍可能被检索到。治法＝上传前 `collection.delete(where={"source": name})`
          清掉同名旧块，代价是「幂等 upsert」变成「删除重建」。
          （真相源那一侧不残：save_chunks 按 source 删旧行再追加，见 rag/ingest.py）
        - 上传是长任务（解析 + 编码，几秒到几十秒），与 /ingest 同一个边界：
          真实生产要改后台任务 + 任务 id 轮询。
    """
    t0 = time.perf_counter()

    # ① 校验：剥掉路径 + 只收 .pdf
    #    为什么必须 Path(...).name：浏览器/脚本可能传 "../../x.pdf" 这类带路径的名字，
    #    直接拿它拼落盘路径就是路径穿越（写到项目目录外面去）。
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail=f"只接受 .pdf 文件，收到 {name!r}")

    # ② 落盘到 UPLOAD_DIR —— 分块流式写，内存占用与文件大小无关。
    #    顺序上先判大小、再落盘：早失败，别白写一个 500MB 的文件才发现超限。
    if file.size and file.size > UPLOAD_MAX_MB * 1024 * 1024:
        # file.size 可能为 None（框架拿不到 Content-Length 时），所以必须先判 None ——
        # 写成 `if file.size > ...` 会在 None 上 TypeError。
        raise HTTPException(
            status_code=400,
            detail=f"文件过大：{file.size / 1024 / 1024:.1f}MB > 上限 {UPLOAD_MAX_MB}MB")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)     # parents=True：data/ 不存在时一并创建
    dst = UPLOAD_DIR / name                           # name 是 ① 校验过的，不含路径分隔符
    with dst.open("wb") as f:
        # file.file 是 SpooledTemporaryFile（同步文件对象）→ 用同步 copyfileobj，不用 await。
        # 默认每块 64KiB：500MB 的扫描件也只占 64KiB 内存。
        # 【不要】改成 `await file.read()`：那是一次性把整个 PDF 读进内存。
        shutil.copyfileobj(file.file, f)

    # ③ 解析切块：`pdf_to_chunks(dst, CHUNK_SIZE)` → 返回 (chunks, pages, skipped)
    #        （rag/splitter.py 里与 scripts/05_split.py 同一份口径，别在这里重写切块逻辑）
    chunks, pages, skipped = pdf_to_chunks(dst, CHUNK_SIZE)

    # ④ 先写【真相源】chunks.jsonl，再写向量库 —— 顺序不能反（Day13 验收后补的修复）。
    #    理由：若先写库、后写 jsonl 而第二步失败，库里就有了「重建追不回」的块，且不报错
    #    （Day13 验收上传的 Redis 笔记 401 块正是这种状态）；反过来失败只是真相源领先库一步，
    #    调 POST /ingest 重建即可对齐。真相源先行的代价最大也就是「多写一次盘」。
    #    传 source=name：重传同名 PDF 时删掉 jsonl 里该 source 的旧行，与向量库
    #    「chunk_id 相同即覆盖」的语义对齐（否则真相源会留下两代同名块，重建时旧块复活）。
    save_chunks(chunks, source=name)

    # ⑤ 入库：`build_index(request.app.state.collection, chunks)` → 返回库里总数
    total = build_index(request.app.state.collection, chunks)

    # ⑥ 组装 UploadResponse；elapsed_ms 用 `(time.perf_counter() - t0) * 1000`
    return UploadResponse(filename=name, pages=pages, skipped=skipped, chunks=len(chunks),
                          total=total, elapsed_ms=(time.perf_counter() - t0) * 1000)


# ============================================================
# Day13 卡 3：知识库文档列表（前端「知识库」标签页的数据源）
# ============================================================
@app.get("/documents", response_model=DocumentsResponse)
def documents(request: Request) -> DocumentsResponse:
    """按 source 聚合向量库，返回「库里有哪些资料」。

    实现要点：
        - 用 `collection.get(include=["metadatas"])` 一次拿全部元数据（932 条，纯内存分组，
          不碰 embedding、不碰 HNSW 检索，几十毫秒级）。不要 `include=["documents"]`：
          正文几十万字，前端根本用不上 —— 与 Day11 硬约束③（出参走白名单）同一条理由。
        - page 为 None 的块（网页来源）不计入 pages；char_count 缺失按 0 计。
        - 排序按块数降序：前端列表第一眼看到的就是资料主体。
    """
    collection = request.app.state.collection
    metas = collection.get(include=["metadatas"])["metadatas"] or []

    agg: dict[str, dict] = {}
    for m in metas:
        src = m.get("source") or "未知来源"
        a = agg.setdefault(src, {"source": src, "chunks": 0, "pages": set(), "chars": 0})
        a["chunks"] += 1
        if m.get("page") is not None:
            a["pages"].add(int(m["page"]))
        a["chars"] += int(m.get("char_count") or 0)

    docs = [
        DocumentItem(source=a["source"], chunks=a["chunks"],
                     pages=len(a["pages"]), chars=a["chars"])
        for a in sorted(agg.values(), key=lambda x: -x["chunks"])
    ]
    return DocumentsResponse(
        collection=collection.name, n_documents=len(docs), documents=docs)


# ============================================================
# Day13：静态前端挂载（必须放在所有路由【之后】）
# ============================================================
# 为什么放最后：Starlette 按注册顺序匹配，`mount("/")` 会吃掉所有未匹配的路径。
# 若它注册在前面，/health /chat /docs 全会被当成静态文件去找 → 404。
# html=True：访问 / 自动返回 index.html（否则只能 /index.html）。
# 同源的好处：前端的 fetch 直接写相对路径 "/chat/stream"，不写死 127.0.0.1:8000，
# 也不依赖 CORS —— 换端口/部署到服务器都不用改前端一行代码。
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    # 等价于命令行 `uvicorn api.main:app --port 8000`；调试时可加 reload=True
    # （注意 reload 会重新走一遍 lifespan，也就是重新加载 reranker，改代码后要等十几秒）
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
