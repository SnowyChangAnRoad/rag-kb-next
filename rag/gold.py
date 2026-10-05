"""数据层 | W1 卡6：页级 gold 生成（expect → gold 集合）。

为什么要它：卡5 的三个指标都是 f(hits, gold)。gold 错了三个一起错，
而且【没有任何一行会报错】—— 空 gold 会被 evaluate 当成「无 gold 题」静默剔除，
分母变小、数字照常打出来，看起来一切正常。这和卡3 第七条（陈旧 token 拿到虚高 IDF）
是同一类失败：**表现是「数字看起来正常」**。
所以本卡真正的产出不是代码，是一份【人工审过、落盘、钉了版本】的 gold。

口径（W1 约定 + 卡6 的一处修订）
------------------------------------------------------------
    in 题          expect 有「第 N 页」且只涉及 1 个文件 → unit="page"
                   gold = 该文件该页的【全部】chunk_id，分母 = 页内块数（3~5）
                   → recall 有 0.2/0.4/0.6/0.8/1.0 五档分辨率，保持计划原文
    cross/partial  涉及多个文件，或没有页码 → unit="source"
                   gold = 期望文件名列表本身，分母 = 文件数（cross = 2）
                   → recall ∈ {0, 0.5, 1}，语义直白：「另一侧证据找到了吗」
    out 题         expect = "库外，应拒答" → gold 为空，被 evaluate 跳过

    ⚠️ unit="source" 是对计划原文的【修订】，理由（实测数字）：
       原文写「cross 题取文件集合 → 该文件全部 chunk 算 gold」，
       而 RAG知识.pdf = 337 块、RAG 技术概览.html = 64 块 → 分母 401，
       recall@5 的天花板是 5/401 ≈ 0.012 —— dense-only 与 hybrid 的差异
       会全部落在小数第三位以后，A/B 直接失去分辨率。
       修订为「命中该文件【任一块】= 该文件命中，分母 = 文件数」后
       recall@5 ∈ {0, 0.5, 1}，才是能看的一列。
       in 题【不改】（页内块级保留五档分辨率）。
       本修订需在《RAG深度强化阶段计划.md》的 W1 口径约定处登记。

判定分支按【数据】走，不按 type 走：type 是人工标的，expect 才是真相。
（本库实测 4 个 source：RAG知识.pdf / Redis全套学习笔记.pdf /
  Transformer--Attention Is All You Need.html / RAG 技术概览.html）

失败必须【显式】（同卡3 第七条的教训）：
    非 out 题抽不到文件 / in 题抽不到页码 / in 题页码在该文件里搜不到任何块
    → 抛 ValueError 并指名题号。绝不静默塌成空 gold。

用法（项目根 rag-kb-next 下，包内代码一律 python -m）：
    python -m rag.gold      # 展开 20 题 → 打印逐题明细（必须人工过一遍）→ 审计 → 落盘
"""

import hashlib
import json
import re
import time
from pathlib import Path

from .config import CHUNKS_JSONL, GOLD_TUNE_JSON, TESTSET_TUNE_JSONL

# 「第 16 页」里的数字。别把模式放宽（如 r"第\d+"）—— expect 里还有「第 4 章」
# 「8GB 显存」这类数字，宽松模式会抓错页码，而且抓错后 gold 会静默变空/变大。
PAGE_RE = re.compile(r"第(\d+)页")

GOLD_VERSION = "gold_tune_v1"


def load_questions() -> list[dict]:
    """已给：读 tune 集 20 题。

    为什么只读 tune 集：frozen_v1.jsonl（10 题）在 W5 前【一行都不许看逐题结果】，
    一旦为了调参看了它，对外数字就不再有意义（计划 §三 门槛 B）。
    """
    with open(TESTSET_TUNE_JSONL, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks() -> list[dict]:
    """已给：读真相源 chunks.jsonl（扁平 schema，没有 meta 子字典 —— 卡3 实测）。"""
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def match_sources(expect: str, sources: list[str]) -> list[str]:
    """TODO-1：expect 提到了哪几个源文件。返回结果保持 sources 的原顺序（确定性）。

    为什么【数据驱动】而不是继续用 eval.py 的 EXPECT_PATTERNS：
        那边手写了 3 条关键词（"RAG知识" / "技术概览" / "Transformer"）——
        源文件改名或新增语料就静默失配，又是一次「不报错的错」。
        而 source 的真实取值就躺在 chunks 里，反推它，将来加语料自动覆盖。

    匹配判据（每个 source 要能吃下它的两种写法）：
        全名：        src in expect
        去掉扩展名：  Path(src).stem in expect      ← expect 里通常不写 .pdf/.html
    本库实测的两种 expect 写法：
        "RAG知识.pdf 第16页"
        "需同时引用 RAG知识.pdf（语义检索）与 RAG 技术概览（向量化）"   → 两个都命中
        "部分覆盖：需引用 RAG 技术概览（嵌入模型微调收益）；8GB 显存微调做法库中无"
    """
    return [
        src
        for src in sources
        if any(candidate in expect for candidate in (src, Path(src).stem))
    ]


def expand_gold(q: dict, chunks: list[dict], sources: list[str]) -> dict:
    """TODO-2：把一道题展开成 gold 记录（本卡核心）。

    输出记录（字段固定，卡7 直接读）：
        {"id", "type", "question", "expect",
         "unit": "page" | "source" | None,
         "gold": [...],        ← unit="page" 时是 chunk_id 列表；"source" 时是文件名列表
         "detail": {"files": [...], "page": int | None, "n_chunks": int}}

        n_chunks 的含义：gold 若按【块】展开本来有多少块（source 粒度下是个审计数字，
        等于这些文件的总块数）。审计时要盯它：
          in 题正常在 1~8（一页 3~5 块）；突然上百 = 页码被抽错或退化成了文件级；
          cross 题应为 337+64=401 —— 这个数字本身就是「为什么不该按块做分母」的证据。

    判定分支（按数据走）：
        files = match_sources(q["expect"], sources)
        m     = PAGE_RE.search(q["expect"])
        page  = int(m.group(1)) if m else None
        if len(files) == 1 and page is not None:      → unit="page"
        else:                                          → unit="source"

        为什么要卡 len(files) == 1：
          cross 题的页码只出现在括号注释里（如「RAG知识.pdf（第75页 检索结果排序与过滤）」），
          那是给人看的说明，不该当硬要求；而且 2 个文件时「第 75 页」只属于其中一个，
          拿它去筛两个文件必然筛空。这和 eval.expect_hit 现有口径一致。

    out 题（type == "out"）：unit=None、gold=[]、detail 里 files=[]/page=None/n_chunks=0。
        evaluate 会把空 gold 的题跳过，所以这里【不报错】—— 拒答题没有 gold 是对的。

    【失败必须显式】三种情况直接 raise ValueError，报错信息里必须带 q["id"]
    （20 题里不写 id 你翻不出是哪道），并把 expect 原文附上便于核对：
        a) type != "out" 但 files 为空              → expect 里没写出任何认识的源文件
        b) type == "in" 但 page 为 None             → in 题必须落到页级；抽不到页码 = expect 写坏了
        c) type == "in" 但按页筛出来 0 块           → 页码在该文件里不存在（或 page 口径不同）。
           ⚠️ 这条最关键：静默返回空 gold 会让这题的 recall 恒为 0，
              指标默默变差、却不报错 —— 属于「必须显式失败」的典型。

    返回：上面那个 dict。gold 用 list（不是 set）—— 落盘要能 json 序列化，
    卡7 读回来再转 set（见文件末尾 load_gold）。
    """
    files = match_sources(q["expect"], sources)
    m = PAGE_RE.search(q["expect"])
    page = int(m.group(1)) if m else None

    # out 题：拒答题没有 gold，evaluate 会跳过它 —— 这里【不报错】是对的
    if q["type"] == "out":
        return {"id": q["id"], "type": q["type"], "question": q["question"],
                "expect": q["expect"], "unit": None, "gold": [],
                "detail": {"files": [], "page": None, "n_chunks": 0}}

    # a) 非 out 题却认不出任何源文件 → 报错（静默空 gold 会让这题被跳过）
    if not files:
        raise ValueError(f"gold.expand_gold: id={q['id']} 认不出源文件，expect={q['expect']!r}")

    # b) in 题必须落到页级：抽不到页码说明 expect 写坏了
    if q["type"] == "in" and page is None:
        raise ValueError(f"gold.expand_gold: id={q['id']} 是 in 题但抽不到「第N页」，"
                         f"expect={q['expect']!r}")

    if len(files) == 1 and page is not None:
        unit = "page"
        gold = [c["chunk_id"] for c in chunks
                if c["source"] == files[0] and c["page"] == page]
        # c) 页码在该文件里搜不到任何块 → 这题 recall 会【恒为 0】，必须在这里拦住
        if not gold:
            raise ValueError(f"gold.expand_gold: id={q['id']} {files[0]} 第{page}页筛出 0 块，"
                             f"页码口径对不上，expect={q['expect']!r}")
        n_chunks = len(gold)
    else:
        unit = "source"                  # cross/partial：卡6 口径修订
        gold = files
        # 这些文件按块展开本来有多少块（审计用）：cross 应为 337+64=401 ——
        # 这个数字本身就是「为什么不该按块做分母」的证据
        n_chunks = sum(1 for c in chunks if c["source"] in files)

    return {"id": q["id"], "type": q["type"], "question": q["question"],
            "expect": q["expect"], "unit": unit, "gold": gold,
            "detail": {"files": files, "page": page, "n_chunks": n_chunks}}


def sha256_file(path) -> str:
    """已给：文件指纹。

    gold 落盘必须钉住【它锚在哪份测试集、哪份语料】上：测试集改了而 gold 没重跑，
    就等于 gold 锚在旧题上 —— 又一种静默失效（同卡3 缓存缺「分词版本」判据那条）。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _audit(records: list[dict]) -> None:
    """TODO-3：把「人工审一遍」变成可执行的断言 —— 人眼会漏，断言不会。

    必写的不变量（每条都要能指出【是哪道题不通过】再 raise）：
        a) 记录了 20 题，id 无重复（tune 集 = in8 / cross5 / partial3 / out4）
        b) out 题（4 道）：unit 为 None 且 gold 为空
        c) in 题（8 道）：unit == "page"，且 detail["n_chunks"] 落在 1~8
           —— 超过 8 几乎一定是页码抽错（退化成文件级就会变成几百）
        d) cross 题（5 道）：unit == "source"，且 len(gold) == 2
           —— 本库 5 道 cross 全是两个文件。将来若真加了单文件 cross 题，
              要【改这条断言】，不要放宽成 >=1 把问题掩盖掉
        e) 所有非 out 题的 gold 非空（空 = gold 塌了，指标会静默变好）

    提示：先收集所有不通过的题号，打印出来再 raise（一次看到全部问题，
    比逐条 assert 停在第 1 条更省时间）。
    """

    problems: list[str] = []

    # a) 记录了 20 题，id 无重复，且 type 分布符合 tune 集：in8 / cross5 / partial3 / out4
    n = len(records)
    if n != 20:
        problems.append(f"总题数应为 20，实际 {n}")

    seen = set()
    dups = set()
    for r in records:
        rid = r.get("id")
        if rid in seen:
            dups.add(rid)
        seen.add(rid)

    if dups:
        problems.append(f"id 重复：{sorted(map(str, dups))}")

    expected_counts = {"in": 8, "cross": 5, "partial": 3, "out": 4}
    type_counts: dict = {}
    for r in records:
        t = r.get("type")
        type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts != expected_counts:
        problems.append(f"type 分布应为 {expected_counts}，实际 {type_counts}")

    for r in records:
        rid = r.get("id", "<无 id>")
        rtype = r.get("type")
        unit = r.get("unit")
        gold = r.get("gold")
        detail = r.get("detail") or {}

        # b) out 题：unit 为 None 且 gold 为空
        if rtype == "out":
            if unit is not None or gold:
                problems.append(
                    f"id={rid} out 题应 unit=None 且 gold=[]，"
                    f"实际 unit={unit!r} gold={gold!r}"
                )
            continue

        # e) 所有非 out 题 gold 非空
        if not gold:
            problems.append(f"id={rid} 非 out 题 gold 为空：{gold!r}")

        # c) in 题：unit == "page"，且 n_chunks 落在 1~8
        if rtype == "in":
            if unit != "page":
                problems.append(f"id={rid} in 题 unit 应为 'page'，实际 {unit!r}")

            n_chunks = detail.get("n_chunks")
            if not isinstance(n_chunks, int) or not (1 <= n_chunks <= 8):
                problems.append(
                    f"id={rid} in 题 n_chunks 应落在 1~8，实际 {n_chunks!r}"
                )

        # d) cross 题：unit == "source"，且 len(gold) == 2
        if rtype == "cross":
            if unit != "source":
                problems.append(
                    f"id={rid} cross 题 unit 应为 'source'，实际 {unit!r}"
                )

            if not isinstance(gold, list) or len(gold) != 2:
                problems.append(
                    f"id={rid} cross 题 gold 应为长度 2 的列表，实际 {gold!r}"
                )

    if problems:
        print("gold 审计未通过：")
        for p in problems:
            print(f"  - {p}")
        raise ValueError(f"gold 审计未通过，共 {len(problems)} 个问题，详见上方打印")


def load_gold() -> dict:
    """已给：读回落盘的 gold，并把 gold 转成 set（卡7 的入口）。

    返回值与落盘结构一致，只是每条记录的 gold 从 list 变成 set，
    可直接喂给 metrics.evaluate 的 rows（它要求 gold 是集合）。
    """
    with open(GOLD_TUNE_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("testset_sha256") != sha256_file(TESTSET_TUNE_JSONL):
        raise RuntimeError(
            f"gold 文件锚定的测试集已变（{payload.get('testset')} "
            f"{payload.get('testset_sha256', '')[:12]}… ≠ 当前 {sha256_file(TESTSET_TUNE_JSONL)[:12]}…）"
            f"—— 重跑 python -m rag.gold，别拿旧 gold 锚新题"
        )
    for r in payload["records"]:
        r["gold"] = set(r["gold"])
    return payload


def main() -> None:
    """已给：展开 → 打印逐题明细（人工过一遍）→ 审计 → 落盘。"""
    questions = load_questions()
    chunks = load_chunks()
    sources = sorted({c["source"] for c in chunks})   # 排序 → 跨运行确定性

    records = [expand_gold(q, chunks, sources) for q in questions]

    print(f"gold 展开：{len(records)} 题 | 真相源 {len(chunks)} 块 | {len(sources)} 个 source")
    print(f"\n{'id':>3} {'type':<8}{'unit':<8}{'页':<5}{'块数':<6}期望文件")
    for r in records:
        d = r["detail"]
        print(f"{r['id']:>3} {r['type']:<8}{str(r['unit']):<8}{str(d['page']):<5}"
              f"{d['n_chunks']:<6}{' + '.join(d['files']) or '-'}")

    _audit(records)

    payload = {
        "version": GOLD_VERSION,
        "testset": TESTSET_TUNE_JSONL.name,
        "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
        "chunks": CHUNKS_JSONL.name,
        "chunks_sha256": sha256_file(CHUNKS_JSONL),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": len(records),
        "records": records,
    }
    GOLD_TUNE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_TUNE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n审计通过，已落盘：{GOLD_TUNE_JSON}")
    print("⚠️ 下一步是【人眼过一遍上面那张表】—— 尤其是 in 题的「页」和「块数」两列，"
          "指标公式对而 gold 错，是所有错误里最难发现的。")


if __name__ == "__main__":
    main()