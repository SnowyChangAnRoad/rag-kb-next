"""裁判层 | Day9：用 LLM 当裁判，量化「答案正确率」与「引用正确率」（本文件无 TODO）。

为什么不再用关键词法（scripts/common_rag.py 的 judge_answer）：
  它只能查「有没有出现拒答词 / 有没有 [n]」这类表面字符串，与语境无关：
    - out 题：'若没有 RAG' 会误命中拒答词「没有」；
    - in 题：只要出现合法 [n] 就算通过 —— Day7 的「全量引用」cites=[1..5] 也能满分。
  两个指标要判断「意思」，只能靠 LLM-as-judge。
  （Day7 的 K4 已经指出：cross 的「来源数≥2」测的是来源分散度，不是是否真的需要综合。）

口径定义（对接阶段总验收第 2 条）：
  答案正确率 = correct 的题数 / 总题数
  引用正确率 = 有效引用条数 / 引用总条数   ← 分母是引用条数，不是题数
               所以「全量标注 [1][2][3][4][5] 但只有 1 条真支撑」会被压到 20%

成本提醒：每题多花 2 次裁判调用（各约 1~1.5k token），
         基线表里的「平均 token / 成本」只统计生成阶段，裁判用量单列。
"""
import json
import re

from .config import LLM_MODEL, THINKING_DISABLED
from .llm import format_context

# ------------------------------------------------------------
# 裁判 1：答案是否被资料支撑（不看引用）
# ------------------------------------------------------------
JUDGE_ANSWER_SYSTEM = """你是严格的事实核查裁判。只判断「回答」是否被「资料」支撑，\
不要评价语气、格式或长度，也不要因为回答标注了引用编号而加分。

判定标准：
- 若资料与问题完全无关（资料本来就答不了这道题），那么回答明确拒答（例如"资料中没有相关内容"）→ correct=true；
- 若资料能回答，回答的核心结论必须能在资料中找到依据，且没有与资料矛盾的内容 → correct=true；
- 出现下列任一情况 → correct=false：编造了资料里没有的事实/数字/结论；答非所问；明明资料能答却拒答。

只输出 JSON：{"correct": true 或 false, "reason": "一句话理由（20 字内）"}"""

# ------------------------------------------------------------
# 裁判 2：逐条核查引用是否真的支撑其所在句子
# ------------------------------------------------------------
JUDGE_CITE_SYSTEM = """你是引用核查裁判。资料以【资料n】编号给出，回答里用 [n] 标注引用。

任务：逐条检查回答里的每一个 [n] —— 该编号对应的资料，是否真的支撑它所在的那句话？

判定标准：
- 支撑：该资料里能找到相关表述，足以推出那句话的结论；
- 不支撑：该资料与那句话无关，或只是沾边但推不出该结论；
  特别注意「全量标注」的情况（回答给几乎每句话都标了 [1][2][3][4][5]），
  这种标注里大部分编号是不支撑的，要如实计入无效。
- 另外：若回答给出了实质结论却完全没有 [n]，missing=true。

只输出 JSON：{"total": 引用总条数, "valid": 其中支撑的条数, "invalid": [不支撑的编号...], \
"missing": true 或 false, "reason": "一句话理由（20 字内）"}"""


def _parse_json(raw: str) -> dict:
    """已给：稳健解析裁判输出（模型偶尔会包 ```json 代码块或前后带解释）。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _call_judge(llm, system: str, user: str) -> tuple[dict, dict]:
    """已给：调一次裁判。temperature=0（裁判要可复现）+ 关闭思考模式。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": 0,
              "extra_body": THINKING_DISABLED}
    try:
        # DeepSeek 的 JSON Output 模式，输出更稳
        resp = llm.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        # 少数网关不支持该参数时退回纯文本，靠 _parse_json 兜底
        resp = llm.chat.completions.create(**kwargs)
    data = _parse_json(resp.choices[0].message.content)
    usage = {"prompt_tokens": resp.usage.prompt_tokens,
             "completion_tokens": resp.usage.completion_tokens}
    return data, usage


def judge_correct(llm, question: str, hits: list[dict], answer: str) -> dict:
    """已给：裁判 1，返回 {"correct": bool, "reason": str, "usage": {...}}。"""
    user = (f"资料：\n{format_context(hits)}\n\n"
            f"问题：{question}\n\n回答：\n{answer}")
    data, usage = _call_judge(llm, JUDGE_ANSWER_SYSTEM, user)
    return {"correct": bool(data.get("correct", False)),
            "reason": str(data.get("reason", "")), "usage": usage}


def judge_citations(llm, question: str, hits: list[dict], answer: str) -> dict:
    """已给：裁判 2，返回 {"total": int, "valid": int, "invalid": [...], "missing": bool, ...}。

    total=0 表示回答没有任何 [n]（此时引用正确率不计入分母，eval 里会单独统计"零引用题数"）。
    """
    user = (f"资料：\n{format_context(hits)}\n\n"
            f"问题：{question}\n\n回答：\n{answer}")
    data, usage = _call_judge(llm, JUDGE_CITE_SYSTEM, user)
    total = int(data.get("total", 0) or 0)
    valid = int(data.get("valid", 0) or 0)
    # 防模型给出 valid > total 的矛盾数字
    valid = max(0, min(valid, total))
    return {"total": total, "valid": valid,
            "invalid": list(data.get("invalid", []) or []),
            "missing": bool(data.get("missing", False)),
            "reason": str(data.get("reason", "")), "usage": usage}