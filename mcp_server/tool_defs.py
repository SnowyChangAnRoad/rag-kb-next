"""W3 卡4：MCP 工具的三个描述档位（由环境变量 RAGKB_DESC_TIER 选档）。

为什么把三档描述集中到这一个文件：卡4 的自变量就是「描述」，三档必须共用同一套
函数体与同一套返回契约，只让描述变 —— 否则因变量就不止一个。

档位（server.py 读 RAGKB_DESC_TIER，默认 full，非法值回退 full）：

  full    （档①）：两个工具，各自的详细 docstring。文案 = B2 定稿，逐字不动。
  minimal （档②）：两个工具，工具名与参数 schema 与档① 完全相同，但函数不写 docstring
                   → description = 空串。
  meta    （档③）：单入口 knowledge_base(query, action, top_k)；描述把两档共性段
                   合并成一份，用 action 在 search / ask 两个实现之间分派。

实验纪律（卡4 结论成立的前提）：
  - 档① 两段文案必须与 B2 快照逐字一致 —— 它是本次对照的基线组。
    共性段（「知识库覆盖」+「能力」）要改就 search / ask 两边同时改。
  - 档③ 只做「分派」，不得把「粗定位 → 精核验」两层压平：
    action=search 与 action=ask 必须是两次独立调用，不能在一次调用里既检索又生成。
  - 返回契约（Source / Hit / SearchResult / AskResult）三档共用。

三处实现口径（都是实测定的，别想当然）：
  ① 描述写在 docstring，不写 description= 参数 —— 卡1 实测 docstring 是描述的真相源，
     两处都写等于给「描述」留两份真相源。
  ② 档② 用 name= 显式对齐工具名：那两个函数为了「没有 docstring」而另起了函数名，
     工具名只能靠参数说明。
  ③ description 的口径：MCPServer 注册的不是 docstring 原始字节 —— 实测
     注册值 = inspect.cleandoc(docstring) + 一个尾换行（档① 注册出来是 550 / 489，
     与 B2 记录一致；raw 是 598 / 533，dedent 是 594 / 529，都对不上）。
     所以「档① 与 B2 基线逐字相同」在证据上体现为：两者 docstring 的 cleandoc 相同。
"""
import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

TIERS = ("full", "minimal", "meta")
DEFAULT_TIER = "full"


# ---------------------------------------------------------------
# 返回契约：给 agent 看的，不是给人看的。字段名也要算 token。
# ---------------------------------------------------------------
class Source(BaseModel):
    """一条来源：让 agent 能区分「哪个文件、哪个块」。

    chunk_id 必需：两个网页来源是整篇级 doc_id，它们的多个块在 page / url 上
    完全相同（实测过 —— 同一网页两条不同块的 Source 逐字段相同）。只有 chunk_id
    能把同一网页的不同块分开，cross 类问题（跨来源取证）才不会在拼来源时塌掉。
    """
    source: str
    chunk_id: str
    page: int | None = None
    url: str | None = None


class Hit(BaseModel):
    """一条命中：来源 + 正文。

    正文不截断的依据是「块长有硬上界」（见 server._check_budget），不是「猜不出哪句重要」。
    """
    text: str
    source: Source


class SearchResult(BaseModel):
    """search 的返回：命中的知识块。

    空命中 = 合法的空列表；不得用「无法回答」这类判断性文本代替（工具只报事实）。
    """
    hits: list[Hit]


class AskResult(BaseModel):
    """ask 的返回：成稿答案 + 结构化来源列表。不回传 hits —— 正文已经进过生成段，
    再回传一次就是同一份内容付第二次上下文费。"""
    answer: str
    sources: list[Source]


class KBResult(BaseModel):
    """档③ 单入口工具的返回：按 action 二选一（search → hits，ask → answer + sources）。"""
    hits: list[Hit] | None = None
    answer: str | None = None
    sources: list[Source] | None = None


# ---------------------------------------------------------------
# 实现体注入点：server.py 启动时把真正的「检索 / 生成」塞进来。
# 不在这里 import server 是为了避开循环导入（server 要 import 本模块）。
# ---------------------------------------------------------------
_SEARCH_IMPL = None      # (query, top_k) -> SearchResult
_ASK_IMPL = None         # (question)     -> AskResult


# ---------------------------------------------------------------
# 档① full：两段文案 = B2 定稿，逐字不动
# ---------------------------------------------------------------
# B0 → B1：自变量就是下面两段 docstring。B0 基线 = 0/6 调用率，两个根因 ——
#   ① 旧描述把触发条件写成「知识不足时调用」= 要求 agent 先自评知识够不够。
#      而这几个问题模型参数里本就有答案 → 它自评「我够用」→ 工具连候选集都进不去。
#      现改为【正向触发条件】（用户说「文档里…」就用它），不再依赖自我诊断。
#   ② 旧描述零「文档」语义锚点，与题面「文档里怎么讲…」的问法对不上。现补覆盖文档映射。
# 另加一条【能力声明】（语义检索；grep 命中不了的改述问题正是本工具的场合）：
#   「语义 vs 关键词」是 rag-kb 相对内置 grep 的唯一结构性优势，不明说，
#   agent 就只会用 Bash + python 去读本地语料。
#
# B2（ask 描述修复）：两条纪律 ——
#   ① 【共性段与 search 逐字相同】（「知识库覆盖」+「能力」两段）。两工具的差异被压缩到
#      只剩「返回什么 / 何时用」一处，于是「ask 是否被选中」就只可能由这一处解释，
#      不会被两段各自演化的样板文字干扰。改共性段必须两边同时改。
#   ② 去掉了上一版的「上述三份文档」—— 悬空指代：工具在列表里各自独立呈现，
#      「上述」没有指代对象（按名字排序时 ask 甚至排在 search 前面）。
#      工具描述必须自包含。
def search_knowledge_base(query: str, top_k: int | None = None) -> SearchResult:
    """语义检索本地 RAG 知识库。返回【原文片段】，不是成稿答案。

    知识库覆盖（只有这三份，别无其他）：
      - RAG知识.pdf                                 RAG 全流程：分块 / 嵌入 / 向量库 / 检索 / 精排 / 生成 / 评估
      - RAG 技术概览.html                            高级 RAG 架构与检索评估
      - Transformer--Attention Is All You Need.html  Transformer 原始论文

    能力：语义检索 —— 问法与原文措辞不同也能命中；
          关键词 grep 命中不了的改述问题，正是本工具最该用的场景。

    何时用：用户提到「文档里 / 资料里 / 论文里 / 知识库里」怎么讲，
            且问题涉及 RAG、向量检索、Transformer 时。
    何时不用：问本地代码 / 日志 / 配置 → 用 grep、Read；问与这三份文档无关的通用知识。
    返回：hit.text + 来源（source / chunk_id / page / url）；无相关内容时返回空列表。
    参数：top_k 默认 3，越大返回越多、上下文成本越高。
    """
    return _SEARCH_IMPL(query, top_k)


def ask_knowledge_base(question: str) -> AskResult:
    """语义检索本地 RAG 知识库。返回【成稿答案】，不是原文片段。

    知识库覆盖（只有这三份，别无其他）：
      - RAG知识.pdf                                 RAG 全流程：分块 / 嵌入 / 向量库 / 检索 / 精排 / 生成 / 评估
      - RAG 技术概览.html                            高级 RAG 架构与检索评估
      - Transformer--Attention Is All You Need.html  Transformer 原始论文

    能力：语义检索 —— 问法与原文措辞不同也能命中；
          关键词 grep 命中不了的改述问题，正是本工具最该用的场景。

    何时用：要的是【一段能直接拿去用的答案】—— 回复给用户、写进文档、
            或把多块内容综合成一段连贯说明时。
    何时不用：要原文片段、要逐字引用、要自己核对出处 → 用 search_knowledge_base。
    返回：answer + sources；文档中无依据时 answer 会说明无法回答。
    """
    return _ASK_IMPL(question)


# ---------------------------------------------------------------
# 档② minimal：同名同 schema，但没有 docstring（description = 空串）
# ---------------------------------------------------------------
def _search_min(query: str, top_k: int | None = None) -> SearchResult:
    return _SEARCH_IMPL(query, top_k)


def _ask_min(question: str) -> AskResult:
    return _ASK_IMPL(question)


# ---------------------------------------------------------------
# 档③ meta：单入口 + action 分派
# ---------------------------------------------------------------
def _dispatch(action: str, query: str, top_k: int | None):
    """按 action 分派到对应实现。

    「分派」不是「合并」：ask 分支不自己去调 search —— 要不要先粗定位、再精核验，
    那个决定权留给 agent（这正是卡4 要观测的行为）。这里只负责把 action 翻译成调用。
    """
    if action == "ask":
        r = _ASK_IMPL(query)
        return KBResult(answer=r.answer, sources=r.sources)
    r = _SEARCH_IMPL(query, top_k)
    return KBResult(hits=r.hits)


def knowledge_base(query: str, action: Literal["search", "ask"],
                   top_k: int | None = None) -> KBResult:
    """语义检索本地 RAG 知识库。按 action 决定返回什么：search → 【原文片段】，ask → 【成稿答案】。

    知识库覆盖（只有这三份，别无其他）：
      - RAG知识.pdf                                 RAG 全流程：分块 / 嵌入 / 向量库 / 检索 / 精排 / 生成 / 评估
      - RAG 技术概览.html                            高级 RAG 架构与检索评估
      - Transformer--Attention Is All You Need.html  Transformer 原始论文

    能力：语义检索 —— 问法与原文措辞不同也能命中；
          关键词 grep 命中不了的改述问题，正是本工具最该用的场景。

    何时用：用户提到「文档里 / 资料里 / 论文里 / 知识库里」怎么讲，
            且问题涉及 RAG、向量检索、Transformer 时。
    action 怎么选：
      search —— 要原文片段、逐字引用、自己核对出处；拿到片段后自己组织语言。
      ask    —— 要一段能直接拿去用的答案：回复用户、写进文档、或多块综合成连贯说明。
    何时不用：问本地代码 / 日志 / 配置 → 用 grep、Read；问与这三份文档无关的通用知识。
    返回：search → hits（text + source / chunk_id / page / url）；无相关内容时是空列表。
          ask    → answer + sources；文档中无依据时 answer 会说明无法回答。
    参数：query 必填；action 必填；top_k 默认 3（仅 search 用），越大返回越多、上下文成本越高。
    """
    return _dispatch(action, query, top_k)


# ---------------------------------------------------------------
# 选档与注册
# ---------------------------------------------------------------
def resolve_tier(raw: str | None) -> str:
    """把环境变量里的档位名规范成一个合法档位；非法/未设 → full（生产默认行为不变）。"""
    tier = (raw or "").strip().lower() or DEFAULT_TIER
    if tier not in TIERS:
        logger.warning("RAGKB_DESC_TIER=%r 不是合法档位，回退 %s（合法值：%s）",
                       raw, DEFAULT_TIER, " / ".join(TIERS))
        return DEFAULT_TIER
    return tier


def register(server, tier: str, search_impl, ask_impl) -> None:
    """按档位把工具注册进 server。search_impl / ask_impl 由 server.py 提供。"""
    global _SEARCH_IMPL, _ASK_IMPL
    _SEARCH_IMPL, _ASK_IMPL = search_impl, ask_impl

    if tier == "minimal":
        server.tool(name="search_knowledge_base")(_search_min)
        server.tool(name="ask_knowledge_base")(_ask_min)
    elif tier == "meta":
        server.tool()(knowledge_base)
    else:
        server.tool()(search_knowledge_base)
        server.tool()(ask_knowledge_base)
