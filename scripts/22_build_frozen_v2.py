"""W5 卡2 | frozen_v2 构建与校验（先写题、后跑分）。

为什么把「校验」单独做成脚本、而不是靠人眼：
    出题阶段的错误全都【不报错】——页码写错会让该题 recall 恒为 0；来源名写错会让 gold
    静默塌成 1 个文件（cross 题的「另一侧」永远算不进来）；id 撞车会让两题共用一份 gold。
    这三类错误肉眼都看不出来，但都能在跑分后被误读成「模型变差了」。
    所以：校验先行，跑分在后。

三条硬纪律（docs/w5_question_spec.md §2.4–2.5）：
    ① 先写题、后跑分：本脚本 --freeze 之后，content 不得再改（改题 = 升版本）；
    ② frozen_v1 只读：构建 v2 时原样并入，只补 kind="fact"，不回写 v1 文件；
    ③ 指纹锚定：gold_frozen_v2.json 记录 frozen_v2.jsonl 的 sha256，
       评测与 pytest 都要校验它，防止「gold 锚在旧题上」的静默失真。

用法（项目根 rag-kb-next 下）：
    python -m scripts.22_build_frozen_v2            # 只校验 + 预览合并结果（不落盘）
    python -m scripts.22_build_frozen_v2 --freeze   # 校验通过后落盘 frozen_v2.jsonl + gold

⚠️ 模块名以数字开头，`python -m` 不能直接运行；实际调用方式是：
    python scripts/22_build_frozen_v2.py [--freeze]
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 允许 `python scripts/22_build_frozen_v2.py` 直接跑（把项目根加进 sys.path）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import (FROZEN_V1_JSONL, FROZEN_V2_DRAFT_JSONL, FROZEN_V2_JSONL,
                        TESTSET_TUNE_JSONL, TEST_JSONL)
from rag.gold import (NO_GOLD_TYPES, PAGE_RE, SETS, build_gold, expand_gold,
                      load_chunks, match_sources, sha256_file)

VALID_TYPES = {"in", "cross", "partial", "out", "principle"}
VALID_KINDS = {"fact", "principle"}
NEW_ID_RANGE = range(31, 51)          # 新增题 id 区间（1–30 已被调参集/冻结集占用）
EXPECT_OUT = "库外，应拒答"
EXPECT_PRINCIPLE = "通用原理，无需检索"

# 题干位置泄漏（spec §2.4 第 7 条）：章节号 / 小节号 / 页码 / 篇号 只允许出现在 expect。
# 为什么必须拦：章节号是 BM25 的精确词项，题干带「7.8 节」等于直接给检索器一条
# gold 附近的强信号，会【人为抬高】recall / MRR，让检索层指标失去真实性。
# 注意不拦「第 3 步 / 4 类 / 四方面」这类普通序数词 —— 只拦指向资料坐标的写法。
LEAK_RE = re.compile(r"(第\s*[0-9一二三四五六七八九十百]+\s*[章节页篇])"
                     r"|([0-9]+(?:\.[0-9]+)+\s*节)")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: str) -> str:
    """去空白 + 去常见中英文标点 —— 防「加个逗号就算新题」的伪去重。"""
    return re.sub(r"[\s，。？！、；：,.?!;:'\"“”‘’（）()【】\[\]-]+", "", text)


def validate_draft(draft: list[dict], others: list[dict], chunks: list[dict],
                   sources: list[str]) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。errors 非空 → 拒绝冻结。"""
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set = set()
    seen_q: dict = {norm(q["question"]): q["id"] for q in others}

    for q in draft:
        qid = q.get("id")
        tag = f"id={qid}"
        # ---- 必填字段 ----
        for field in ("id", "type", "split", "version", "created_at", "kind",
                      "question", "expect"):
            if field not in q or q[field] in ("", None):
                errors.append(f"{tag} 缺字段 {field!r}")
        if errors and any(tag in e for e in errors):
            continue

        if not isinstance(qid, int) or qid not in NEW_ID_RANGE:
            errors.append(f"{tag} 新增题 id 必须落在 {NEW_ID_RANGE.start}–{NEW_ID_RANGE.stop - 1}")
        if qid in seen_ids:
            errors.append(f"{tag} id 重复")
        seen_ids.add(qid)

        qtype, kind = q["type"], q["kind"]
        if qtype not in VALID_TYPES:
            errors.append(f"{tag} type={qtype!r} 不在 {sorted(VALID_TYPES)}")
            continue
        if kind not in VALID_KINDS:
            errors.append(f"{tag} kind={kind!r} 不在 {sorted(VALID_KINDS)}")
        if (qtype == "principle") != (kind == "principle"):
            errors.append(f"{tag} type 与 kind 不一致：type={qtype!r} kind={kind!r}"
                          f"（principle 必须成对出现）")

        # ---- 题干去重（与既有 30 题 + 本次草稿） ----
        nq = norm(q["question"])
        if nq in seen_q:
            errors.append(f"{tag} 题干与 id={seen_q[nq]} 重复（规范化后完全相同）")
        else:
            seen_q[nq] = qid

        # ---- 题干禁位置泄漏（spec §2.4 第 7 条）----
        leak = LEAK_RE.search(q["question"])
        if leak:
            errors.append(f"{tag} 题干含资料位置 {leak.group(0)!r}（位置只允许写进 expect）")

        # ---- expect 白名单 + 语料可答性 ----
        expect = q["expect"]
        files = match_sources(expect, sources)
        page_m = PAGE_RE.search(expect)

        if qtype == "in":
            if page_m is None:
                errors.append(f"{tag} in 题 expect 缺「第N页」：{expect!r}")
            if len(files) != 1:
                errors.append(f"{tag} in 题 expect 必须恰好命中 1 个来源，实际 {files}")
            if page_m and len(files) == 1:
                page = int(page_m.group(1))
                n = sum(1 for c in chunks if c["source"] == files[0] and c["page"] == page)
                if n == 0:
                    errors.append(f"{tag} {files[0]} 第{page}页在语料里筛出 0 块")
                elif n > 8:
                    warnings.append(f"{tag} 第{page}页有 {n} 块（>8，先确认页码没抽错）")

        elif qtype == "cross":
            if len(files) != 2 or len(set(files)) != 2:
                errors.append(f"{tag} cross 题必须恰好命中 2 个不同来源，实际 {files}")
            if page_m and len(files) == 1:
                errors.append(f"{tag} cross 题的页码只能写进括号注释，不能当硬锚点")

        elif qtype == "partial":
            if len(files) != 1:
                errors.append(f"{tag} partial 题必须恰好命中 1 个来源（库内那一半），实际 {files}")
            if "库中无" not in expect and "库里无" not in expect:
                errors.append(f"{tag} partial 题 expect 必须显式写出「库中无」的部分：{expect!r}")

        elif qtype == "out":
            if expect != EXPECT_OUT:
                errors.append(f"{tag} out 题 expect 必须逐字为 {EXPECT_OUT!r}")
            if files:
                errors.append(f"{tag} out 题 expect 不应命中任何库内来源，实际 {files}")

        elif qtype == "principle":
            if expect != EXPECT_PRINCIPLE:
                errors.append(f"{tag} principle 题 expect 必须逐字为 {EXPECT_PRINCIPLE!r}")
            if files:
                errors.append(f"{tag} principle 题不应命中库内来源，实际 {files}")
            ref = q.get("reference")
            if not isinstance(ref, list) or len(ref) < 2:
                errors.append(f"{tag} principle 题必须有 reference 列表（≥2 条要点）")

    return errors, warnings


def merge_with_v1(draft: list[dict], v1: list[dict]) -> list[dict]:
    """frozen_v1 原样并入，只补 kind='fact'（源文件不动 —— 升版本纪律）。"""
    merged = []
    for q in v1:
        if "kind" in q:
            raise ValueError(f"frozen_v1 id={q['id']} 已自带 kind —— v1 应保持只读，勿手工改")
        merged.append({**q, "kind": "fact"})
    return sorted(merged + draft, key=lambda q: q["id"])


def check_merged(merged: list[dict], chunks: list[dict], sources: list[str],
                 counts: dict) -> list[str]:
    """合并后再走一遍 gold 展开 —— 直接用 gold 的硬断言，避免两套口径。"""
    errors: list[str] = []
    actual: dict = {}
    for q in merged:
        actual[q["type"]] = actual.get(q["type"], 0) + 1
    if actual != counts:
        errors.append(f"分层配比应为 {counts}，实际 {actual}")

    for q in merged:
        try:
            expand_gold(q, chunks, sources)
        except ValueError as e:
            errors.append(str(e))
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="校验通过后落盘 frozen_v2.jsonl 与 gold（不可覆盖已存在的冻结文件）")
    ap.add_argument("--draft", default=str(FROZEN_V2_DRAFT_JSONL))
    args = ap.parse_args()

    draft_path = Path(args.draft)
    counts = SETS["frozen_v2"]["counts"]

    draft = load_jsonl(draft_path)
    v1 = load_jsonl(FROZEN_V1_JSONL)
    others = load_jsonl(TEST_JSONL) + load_jsonl(TESTSET_TUNE_JSONL)
    chunks = load_chunks()
    sources = sorted({c["source"] for c in chunks})

    print(f"草稿 {len(draft)} 题 | frozen_v1 {len(v1)} 题 | 既有题 {len(others)} 题 | "
          f"语料 {len(chunks)} 块 / {len(sources)} 个 source")

    errors, warnings = validate_draft(draft, others, chunks, sources)
    merged = merge_with_v1(draft, v1)
    errors += check_merged(merged, chunks, sources, counts)

    for w in warnings:
        print(f"  ⚠️ {w}")
    if errors:
        print(f"\n校验未通过（{len(errors)} 个问题）：")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("\n✅ 校验通过")

    print(f"\n合并后 {len(merged)} 题：")
    print(f"{'id':>3} {'type':<10}{'kind':<10}{'v':<8}题干 / expect")
    for q in merged:
        ver = "v1" if q["id"] < 31 else "v2-new"
        print(f"{q['id']:>3} {q['type']:<10}{q['kind']:<10}{ver:<8}"
              f"{q['question'][:38]}｜{q['expect'][:42]}")

    if not args.freeze:
        print("\n（预览模式，未落盘。审题通过后加 --freeze）")
        return

    if FROZEN_V2_JSONL.exists():
        raise SystemExit(f"{FROZEN_V2_JSONL} 已存在 —— 冻结文件不可覆盖（改题 = 升版本），"
                         f"请先确认是否真的要升 v3")
    with open(FROZEN_V2_JSONL, "w", encoding="utf-8") as f:
        for q in merged:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    sha = sha256_file(FROZEN_V2_JSONL)
    print(f"\n已冻结：{FROZEN_V2_JSONL}\nsha256 = {sha}")

    payload = build_gold("frozen_v2")
    print(f"gold 锚定 sha256 = {payload['testset_sha256']}")
    if payload["testset_sha256"] != sha:
        raise SystemExit("gold 指纹与刚冻结的文件不一致 —— 落盘期间文件被改动了")


if __name__ == "__main__":
    main()
