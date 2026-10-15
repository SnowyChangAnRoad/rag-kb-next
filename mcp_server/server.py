"""MCP server | W3 卡2 建、卡4 扩：把 rag-kb 知识库暴露为 tool，供外部 Agent 调用。

传输：stdio（本地子进程）—— Claude Code / MCP Inspector 走的都是这条。

运行（必须在项目根目录；`-m` 会把 cwd 加进 sys.path，`rag.*` 才可导入）：
    & "C:\\Users\\xx\\Desktop\\python\\.venv\\Scripts\\python.exe" -m mcp_server.server

卡4：工具的「描述」有三档，由环境变量 RAGKB_DESC_TIER 选（默认 full）。三档的目标、
文案与实验纪律全写在 tool_defs.py；本文件只做两件事 ——
    ① 提供「检索 / 生成」两个实现（返回值必须与 tool_defs 里的契约模型对齐）
    ② 按档位把它们注册成工具
    full / minimal：search_knowledge_base + ask_knowledge_base
    meta          ：knowledge_base(query, action, top_k)

已定的设计口径（卡2 设计草案，见计划文档 W3 记录）：
    search_knowledge_base ：召回 → 精排 → 去重 → 阈值 → 截断
    ask_knowledge_base    ：完整 pipeline（检索 + 生成）
    相关性阈值            ：两工具都传 min_score=0.05，砍掉「检索到但明显不相关」的块
    空命中                ：search 返回空列表（只报事实）；「答不了」由 ask 的生成段
                            按 prompt 规则决定，不在工具逻辑里
    返回形态              ：结构化（structured_output），从返回类型注解生成 output_schema
"""
import logging
import os
import sys
from pathlib import Path

# 兜底：若用 `python mcp_server/server.py` 直接跑，sys.path[0] 是 mcp_server/、
# 项目根不在其中 → `rag.*` 导入即失败。这里显式补上（走 `-m` 时其实已有，留着无害）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer

# 返回契约与三档描述都在 tool_defs（server ↔ tool_defs 单向依赖：tool_defs 不 import 本模块）
from mcp_server import tool_defs
from mcp_server.tool_defs import AskResult, SearchResult

from rag.config import RECALL_K, RERANK_TOP_K
from rag.pipeline import RAGPipeline
from rag.retriever import get_client, get_collection
from rag.trace import nullspan

server = MCPServer("rag-kb")

logger = logging.getLogger(__name__)

# 常驻单例：932 块库 + reranker 权重只加载一次。
# stdio server 是长驻进程，放模块级即可；绝不能写进 tool 函数体——那会变成
# 「每次调用都重载模型」，每次多几十秒。
_pipeline = RAGPipeline(get_collection(get_client()))


# ---------------------------------------------------------------
# 体积预算
# ---------------------------------------------------------------
# 事实：正文不截断之所以安全，是因为块长有硬上界 —— config.CHUNK_SIZE=300，
#       实测 max char_count=330（含 10% overlap）。返回体积 ≈ 条数 × 单条字符数。
#       一旦有人改大 CHUNK_SIZE，或语料换成大块文档，这里必须能响。
# 实测（RECALL_K=10、去重后 9 条）：total_chars 2673 / 2396，单条 max 325 —— 余量约 19%。
# 所以这张网不是装饰品：块长口径再涨 ~23% 就会真的触发。
MAX_RESULT_CHARS = 3300      # 名字对齐语义：这是「一次返回的总预算」= RECALL_K × 单块上界


def _check_budget(hits: list[dict]) -> list[dict]:
    """按预算校验一次返回的体积：超限则丢弃后面的命中，但不许静默。

    这张网要拦的是「块长口径被改了」。静默截断会把「口径变了」这个开发期事实
    转成运行期的数据丢失 —— 网还在，却不再出声，等于没有。
    """
    total_chars = 0
    for i, h in enumerate(hits):
        total_chars += len(h["text"])
        if total_chars > MAX_RESULT_CHARS:
            logger.warning(
                "search_knowledge_base 返回体积超预算：丢弃后 %d 条"
                "（预算 %d 字符 / 累计 %d / 命中共 %d 条）。先查 CHUNK_SIZE 与块长口径。",
                len(hits) - i, MAX_RESULT_CHARS, total_chars, len(hits))
            return hits[:i]
    return hits


def _clamp_top_k(top_k: int | None) -> int:
    """把调用方给的 top_k 夹进 [1, RECALL_K]；None → RERANK_TOP_K（服务默认口径）。

    上限取 RECALL_K 而不是另立常量：精排池的深度就是 RECALL_K，想要更多条只能先
    加深召回池 —— 那是另一个变量，不在本工具的契约里。
    """
    if top_k is None:
        return RERANK_TOP_K
    return max(1, min(top_k, RECALL_K))


# ---------------------------------------------------------------
# 管道：检索与字段映射（机械部分，与 ask 共用同一段检索）
# ---------------------------------------------------------------
# 相关性阈值（卡2 实测定的口径）：库外问题 top1 ≤ 0.01，库内 top1 ≥ 0.2553，间隙 25 倍。
# 取 0.05 = 对库外上界的 5 倍余量。_retrieve 默认 0.0，所以评测基线不受影响。
# ⚠️ 样本只有 6 题 —— 卡7 必须在 20 题上回归，否则就是过拟合。
MIN_SCORE = 0.05


def _retrieve(query: str, top_k: int) -> list[dict]:
    """召回 → 精排 → 去重 → 截断。

    为什么直接调 `pipeline._retrieve` 而不是自己拼 hybrid_retrieve + ranker：
        W1 的纪律是「服务路径与实验脚本共用除召回段以外的全部代码」。这里若另起
        一条检索链，卡7 的 A/B 结论就不再适用于服务路径 —— 那个 A/B 的前提正是
        「两组只有召回段不同」。
    """
    hits, _recall, _ranked, _recall_ms, _rerank_ms, _k, _mode = _pipeline._retrieve(
        query, True, top_k, None, nullspan, min_score=MIN_SCORE)
    return hits


# 进 Source 的原始字段：按上面 Source 的定义增删（这里只做字段挑选，不做加工）。
_SOURCE_FIELDS = ("source", "chunk_id", "page", "url")


def _to_sources(hits: list[dict]) -> list[dict]:
    """hits → 结构化来源列表。

    返回裸 dict 即可：pydantic 会在 SearchResult / AskResult 里按注解把它校成 Source，
    于是「哪些字段进契约」只有 Source 一处真相源，不必在这里再抄一份。
    """
    return [{k: h.get(k) for k in _SOURCE_FIELDS} for h in hits]


# ---------------------------------------------------------------
# 两个实现 + 选档注册（描述本体在 tool_defs.py）
# ---------------------------------------------------------------
def _search_impl(query: str, top_k: int | None = None) -> SearchResult:
    """召回 → 精排 → 去重 → 阈值 → 截断。"""
    k = _clamp_top_k(top_k)
    hits = _check_budget(_retrieve(query, k))     # 返回值必须接住，否则截断等于没写
    return SearchResult(hits=[{"text": h["text"], "source": s}
                              for h, s in zip(hits, _to_sources(hits))])


def _ask_impl(question: str) -> AskResult:
    """完整 pipeline（检索 + 生成）。

    这里不查体积预算：ask 的返回体只有 answer + 来源，很小。
    要限的是「进 prompt 的检索体积」，那已经由 pipeline 内部的 top_k 控制住了。

    用 prompt_version="v1-sec"（卡5 注入加固版）：服务路径默认走加固版，
    V1 冻结保留给 Day10 的拒答实验口径。加固理由与实测见 rag.config 该常量注释。
    """
    out = _pipeline.answer(question, min_score=MIN_SCORE, prompt_version="v1-sec")
    return AskResult(answer=out["answer"], sources=_to_sources(out["hits"]))


# 档位在进程启动时定死：stdio server 是长驻子进程，一个进程只注册一套工具。
# 换档位 = 改 RAGKB_DESC_TIER → 重启 Claude Code（MCP server 由它拉起）。
tier = tool_defs.resolve_tier(os.environ.get("RAGKB_DESC_TIER"))
tool_defs.register(server, tier, search_impl=_search_impl, ask_impl=_ask_impl)

# 打 stderr：stdio 下 stdout 只能走 MCP 协议，不能拿来打日志。
# 档位必须留痕，否则「这一跑到底是哪一档」只能靠猜。
print(f"[rag-kb] 描述档位 RAGKB_DESC_TIER={tier}", file=sys.stderr)


if __name__ == "__main__":
    server.run()      # 默认 stdio；不要传 transport="http"——那是另一张卡的事