"""生成层 | Day9：DeepSeek 调用、prompt 拼装、token/成本统计。

分层原则：本模块只管「把 hits 变成一次回答」，不碰检索、不碰裁判。

用法：
    python -m rag.llm probe        Day12 探针：打印 stream=True 的原始 chunk 结构
"""
import os

from collections.abc import Iterator

from openai import OpenAI

from .config import (
    LLM_MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M, PROMPTS, TEMPERATURE,
    THINKING_DISABLED, USER_TEMPLATE,
)


def get_llm() -> OpenAI:
    """已给：OpenAI 兼容 SDK 连 DeepSeek，key / base_url 从 .env 读。"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


# ------------------------------------------------------------
# 拼装（生成与裁判共用，避免两处格式漂移）
# ------------------------------------------------------------
def safe_preview(text: str, n: int = 40) -> str:
    """已给：控制台短预览，避免特殊 Unicode 触发 cp936 编码错误（Day3 教训）。"""
    return text.replace("\n", " ")[:n]


def hit_where(hit: dict) -> str:
    """已给：统一「第x页」/ url 的定位文案（PDF 用 page，网页用 url）。"""
    if hit.get("page") is not None:
        return f"第{hit['page']}页"
    return hit.get("url") or "网络资料"


def format_context(hits: list[dict]) -> str:
    """已给：把 hits 拼成【资料n】块。回答里的 [n] 就指向这里的 n，裁判也读同一份。"""
    blocks = [
        f"【资料{i}】来源：{h['source']} {hit_where(h)}\n{h['text']}"
        for i, h in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict]) -> str:
    """已给：资料 + 问题，拼成完整的 user 内容。"""
    return USER_TEMPLATE.format(context=format_context(hits), question=question)


def format_sources(hits: list[dict]) -> str:
    """已给：来源列表，每行形如 [1] RAG知识.pdf 第15页。"""
    lines = [f"[{i}] {h['source']} {hit_where(h)}" for i, h in enumerate(hits, 1)]
    return "\n".join(lines) + "\n" if lines else ""


# ------------------------------------------------------------
# 流式探针（Day12 卡 1）—— 先看原始数据，再写解析代码
# ------------------------------------------------------------
def probe_stream(question: str = "用一句话说明什么是向量检索") -> None:
    """证据先行：把 `stream=True` 的原始 chunk 打出来，再动手写 ask_stream。

    为什么必须先探针（Day8「先诊断后动手」的同一方法）：
        流式返回的结构与一次性调用【不是】同一套 —— 正文藏在 delta 里而不是 content；
        「token 用量」这种东西只在流的最后才出现，甚至可能不出现；
        而带 usage 的那一个 chunk，它的 choices 很可能是【空列表】。
        靠文档猜字段名，就会写出 `chunk.choices[0]` 直接 IndexError 的代码。
        花 1 分钱把真实结构打出来，比读十页文档都靠谱。

    成本：两次极短调用，约 0.001 元。

    用法（项目根 rag-kb 下，需关闭沙箱）：
        python -m rag.llm probe
    """
    llm = get_llm()
    cases = [
        ("① 默认：不带 stream_options", dict(THINKING_DISABLED), {}),
        ("② 带 stream_options.include_usage",
         dict(THINKING_DISABLED), {"stream_options": {"include_usage": True}}),
    ]
    for label, extra_body, kwargs in cases:
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        try:
            stream = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=TEMPERATURE, stream=True,
                extra_body=extra_body, **kwargs,
            )
        except Exception as e:                     # 探针的职责就是「看清楚它到底支不支持」
            print(f"调用直接失败：{type(e).__name__}: {e}")
            continue

        n_text, usage_at, empty_choices, last = 0, None, [], -1
        for i, chunk in enumerate(stream):
            last = i
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_at = i
            if not chunk.choices:                  # ← 关键观察点：带 usage 的 chunk choices 为空
                empty_choices.append(i)
                print(f"#{i:>3}  choices=0  usage={usage}")
                continue
            delta = chunk.choices[0].delta
            text = delta.content or ""
            n_text += bool(text)
            if i < 4 or usage is not None:
                print(f"#{i:>3}  choices={len(chunk.choices)} "
                      f"finish={chunk.choices[0].finish_reason!r} "
                      f"delta={text[:40]!r} "
                      f"reasoning_content={bool(getattr(delta, 'reasoning_content', None))}"
                      + (f"  usage={usage}" if usage is not None else ""))
        print(f"--- 小结：共 {last + 1} 个 chunk，其中带正文的 {n_text} 个；"
              f"choices 为空的 chunk 下标={empty_choices}；usage 出现在 #{usage_at}")


# ------------------------------------------------------------
# 一次生成
# ------------------------------------------------------------
def build_messages(question: str, hits: list[dict], system_prompt: str) -> list[dict]:
    """一次问答的 messages —— ask() 与 ask_stream() 共用。

    单抽这一层不是为了少写两行，是为了让「流式/非流式走同一个 prompt」成为结构保证：
    Day9 已吃过 format_context 两处实现漂移的亏，prompt 结构一旦漂移，
    两个入口的实验结果就不可比 —— 而 Day14 的最终指标正是两边都要读的。
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": build_prompt(question, hits)},
    ]


def ask(llm: OpenAI, question: str, hits: list[dict],
        system_prompt: str = PROMPTS["v1"]) -> dict:
    """调一次 DeepSeek，返回结构化生成结果。

    返回字段（下游 pipeline / eval / Day10 埋点都要用）：
        {"answer": str, "prompt_tokens": int, "completion_tokens": int, "cost": float}

    system_prompt 由 pipeline 按 prompt_version 解析后传进来（Day10 卡 4：一张测试集要能
    跑多个 prompt 版本，所以这里不再直接读全局常量）。
    """
    resp = llm.chat.completions.create(
        model=LLM_MODEL, messages=build_messages(question, hits, system_prompt),
        temperature=TEMPERATURE, extra_body=THINKING_DISABLED,
    )
    answer = resp.choices[0].message.content
    usage = resp.usage
    cost = usage.prompt_tokens / 1e6 * PRICE_IN_PER_M + usage.completion_tokens / 1e6 * PRICE_OUT_PER_M

    return {"answer": answer, "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens, "cost": cost}


def ask_stream(llm: OpenAI, question: str, hits: list[dict],
               system_prompt: str = PROMPTS["v1"]) -> Iterator[dict]:
    """流式版 ask()：边收边吐（Day12）。

    yield 两种事件，都是【纯数据 dict，不含任何 SSE 概念】—— event:/data: 是 HTTP 层的语言，
    生成层不该认识它。这样 Day13 换前端、Day14 换评测入口，本函数一行都不用动。

        {"type": "delta", "text": "增量的几个字"}      收到即可渲染
        {"type": "done",  "answer": 完整答案, "prompt_tokens":..., "completion_tokens":..., "cost":...}

    三个必须处理的点（全部来自 probe_stream 的实测，不是靠文档猜的）：

      ① delta 是【增量】不是全文。必须自己累加 parts 才能得到完整答案 ——
         trace 落盘、评测判引用、指标统计读的都是完整答案，漏了这步这些功能会【静默】失效
         （HTTP 那头照样能拼出完整回答，所以 bug 不会当场暴露）。

      ② usage 只在最后一个 chunk 出现，必须边流边记 —— 流迭代完就再也拿不到了。
         实测：DeepSeek 不传 stream_options 也会在末 chunk 给 usage（和 finish_reason 合并在一起）；
         而 OpenAI 官方是「不传就永不给」。这里仍显式传 stream_options.include_usage：
         靠未文档化的兼容行为是脆的，多传一个参数的代价是零。

      ③ 仍要判空 choices。实测本版本末 chunk 的 choices 非空，但「带 usage 的 chunk choices 为空」
         是 OpenAI 兼容生态里的常见形态 —— 不判空就是一行现成的 IndexError，
         且它只会在换网关时才出现，属于最难查的那类错。

    usage 拿不到时（理论分支，换到不给 usage 的网关会走到）：token/成本返回 None，
    而不是填 0 —— 0 会被下游当成真实数据统计（Day11 的 chunks:0 教训同一个道理）。
    """
    stream = llm.chat.completions.create(
        model=LLM_MODEL, messages=build_messages(question, hits, system_prompt),
        temperature=TEMPERATURE, stream=True,
        stream_options={"include_usage": True}, extra_body=THINKING_DISABLED,
    )
    parts: list[str] = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:              # 见 ③
            continue
        text = chunk.choices[0].delta.content
        if text:                           # 首 chunk 的 delta 是空串（只带 role），不必往下传
            parts.append(text)
            yield {"type": "delta", "text": text}

    pt = usage.prompt_tokens if usage else None
    ct = usage.completion_tokens if usage else None
    yield {
        "type": "done", "answer": "".join(parts),
        "prompt_tokens": pt, "completion_tokens": ct,
        "cost": (pt / 1e6 * PRICE_IN_PER_M + ct / 1e6 * PRICE_OUT_PER_M) if usage else None,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe_stream(sys.argv[2] if len(sys.argv) > 2 else "用一句话说明什么是向量检索")
    else:
        print(__doc__)