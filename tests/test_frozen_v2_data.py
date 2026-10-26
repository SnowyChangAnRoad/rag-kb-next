"""W5 卡5 | 冻结集的数据不变量（零 API、毫秒级）。

为什么这些断言值得写：它们拦的全是「不报错的错」——
    题干里又混进章节号（检索泄漏）、gold 静默塌空、gold 锚在旧题上、
    principle 题缺 reference 导致判分无法执行。
这些错误在跑分时都不会抛异常，只会让数字变得好看或难看。

运行：
    & "<venv>\\Scripts\\python.exe" -m pytest tests/test_frozen_v2_data.py -q
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import (  # noqa: E402
    CHUNKS_JSONL, FROZEN_V2_BASELINE_JSON, FROZEN_V2_JSONL, GOLD_FROZEN_V2_JSON,
    RECALL_K, RERANK_TOP_K, RRF_K, SOURCE_CAP,
)
from rag.gold import SETS, sha256_file  # noqa: E402

# 与 FROZEN_V2_DRAFT 校验脚本同一套正则（spec §2.4 第 7 条）
LEAK_RE = re.compile(r"(第\s*[0-9一二三四五六七八九十百]+\s*[章节页篇])"
                     r"|([0-9]+(?:\.[0-9]+)+\s*节)")
NO_GOLD_TYPES = ("out", "principle")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return load_jsonl(FROZEN_V2_JSONL)


@pytest.fixture(scope="module")
def gold() -> dict:
    with open(GOLD_FROZEN_V2_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return load_jsonl(CHUNKS_JSONL)


def test_layout_matches_spec(questions):
    """30 题、分层配比与 spec §2.1 一致、id 唯一、kind 与 type 成对。"""
    assert len(questions) == 30
    counts: dict = {}
    for q in questions:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    assert counts == SETS["frozen_v2"]["counts"], f"分层配比漂了：{counts}"
    assert len({q["id"] for q in questions}) == 30, "id 有重复"
    for q in questions:
        assert q["kind"] in ("fact", "principle"), f"id={q['id']} kind 非法"
        assert (q["type"] == "principle") == (q["kind"] == "principle"), \
            f"id={q['id']} type/kind 不成对"


def test_question_has_no_position_leak(questions):
    """题干不得出现章节号/页码（位置泄漏会人为抬高 recall，spec §2.4 第 7 条）。"""
    bad = [(q["id"], LEAK_RE.search(q["question"]).group(0))
           for q in questions if LEAK_RE.search(q["question"])]
    assert not bad, f"题干含资料位置（应只写在 expect 里）：{bad}"


def test_gold_is_anchored_to_current_files(gold):
    """gold 必须锚在当前这份 frozen_v2 与 chunks 上 —— 否则指标静默失真。"""
    assert gold["testset_sha256"] == sha256_file(FROZEN_V2_JSONL), \
        "gold 锚定的测试集已变：重跑 python -m rag.gold frozen_v2"
    assert gold["chunks_sha256"] == sha256_file(CHUNKS_JSONL), \
        "gold 锚定的语料已变：重跑 python -m rag.gold frozen_v2"


def test_gold_records_invariants(gold, chunks):
    """逐题 gold 不变量：无 gold 题必须真空，有 gold 题必须真有。"""
    chunk_ids_by_page: dict = {}
    for c in chunks:
        chunk_ids_by_page.setdefault((c["source"], c["page"]), set()).add(c["chunk_id"])

    for r in gold["records"]:
        rid, rtype, unit, g = r["id"], r["type"], r["unit"], r["gold"]
        if rtype in NO_GOLD_TYPES:
            assert unit is None and g == [], f"id={rid} {rtype} 应 unit=None gold=[]"
            continue
        assert g, f"id={rid} 非 out/principle 题 gold 为空 → 该题 recall 恒 0"
        if rtype == "in":
            assert unit == "page", f"id={rid} in 题 unit 应为 page"
            page = r["detail"]["page"]
            src = r["detail"]["files"][0]
            assert set(g) == chunk_ids_by_page[(src, page)], \
                f"id={rid} gold 与「{src} 第{page}页」的块集合不一致"
        elif rtype == "cross":
            assert unit == "source" and len(g) == 2, f"id={rid} cross 题应 2 个来源"
        else:                                    # partial
            assert unit in ("page", "source") and g, f"id={rid} partial gold 非法"


def test_principle_questions_have_reference(questions):
    """principle 题必须带 reference —— 没有要点，judge 无从判覆盖率。"""
    for q in questions:
        if q["type"] != "principle":
            continue
        ref = q.get("reference")
        assert isinstance(ref, list) and len(ref) >= 2, \
            f"id={q['id']} principle 题 reference 不足 2 条"


def test_baseline_matches_config_and_testset():
    """基线必须锚在同一份题、同一组检索参数上；参数改了而基线没升 = 门禁名存实亡。"""
    if not FROZEN_V2_BASELINE_JSON.exists():
        pytest.skip("基线尚未生成（先跑 scripts/23_eval_frozen_v2.py）")
    with open(FROZEN_V2_BASELINE_JSON, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    assert baseline["frozen_v2_sha256"] == sha256_file(FROZEN_V2_JSONL), \
        "题目已变而基线没升版本 —— 基线结论不再适用"
    fp = baseline["config_fingerprint"]
    assert fp["recall_k"] == RECALL_K and fp["rerank_top_k"] == RERANK_TOP_K
    assert fp["rrf_k"] == RRF_K and fp["source_cap"] == SOURCE_CAP
    assert baseline["gate"]["tolerance"] == 0.0, "门禁口径必须是严格全等"
