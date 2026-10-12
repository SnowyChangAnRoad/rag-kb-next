"""实验 | W2 卡3 验收⑥：dense 路径回归（证明改了 retrieve() 后 day6_kb 结果逐位不变）。

为什么必须有这一条
    卡3 把查询编码从「Chroma 内部替我们调 embedding_function」改成「我们显式
    encode_query + query_embeddings=」。数学上应当是同一条路径（BGE 的 query 与 doc
    本来就是同一个函数、同一套归一化），但「应当」不算证据：显式传向量时若给了
    numpy float32（而原路径是 Python list[float]），末位就可能不同。
    所以按项目硬规则：dense 组【整组重跑】，与卡7 落盘的 data/eval_w1_ab.json
    逐字段用 != 比较 —— 不许用 isclose，isclose 会把「已经变成另一个数」判成合格。

口径对齐（与卡7 的 retrieve_stage 完全一致，别加别减）
    召回 hybrid_retrieve(..., mode="dense") → 精排 rerank(全量降序) → 去重 dedupe_adjacent
    证据层 = post_dedupe_pre_quota（未配额、未截断）

比较方式说明
    证据里的 rerank_score 是 round(..., 4) 落盘的，所以这边也比 round(..., 4)，
    再逐位 != —— 仍是精确比较，只是把比较基准对齐到落盘精度。

用法（项目根 rag-kb-next 下）
    & "C:\\Users\\xx\\Desktop\\python\\.venv\\Scripts\\python.exe" scripts/15_regress_dense.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import GOLD_TUNE_JSON, RECALL_K  # noqa: E402
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

EVIDENCE_JSON = ROOT / "data" / "eval_w1_ab.json"
MODE = "dense"


def dense_stage(collection, ranker, question: str) -> list[dict]:
    """卡7 retrieve_stage 的 dense 分支：召回 → 精排 → 去重（未配额、未截断）。"""
    hits = hybrid_retrieve(collection, question, RECALL_K, mode=MODE)
    return dedupe_adjacent(ranker.rerank(question, hits))


def main() -> None:
    with open(EVIDENCE_JSON, "r", encoding="utf-8") as f:
        evidence = json.load(f)
    with open(GOLD_TUNE_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)["records"]

    qmap = {r["id"]: (r["question"], r["type"]) for r in records}
    exp = {r["id"]: r for r in evidence["per_question"][MODE]}

    print(f"回归：{EVIDENCE_JSON.name}（{evidence['generated_at']}）| 题数={len(exp)} "
          f"| RECALL_K={RECALL_K} | mode={MODE}")
    print(f"证据指纹：testset={evidence['fingerprints']['testset_sha256'][:12]}… "
          f"gold={evidence['fingerprints']['gold_sha256'][:12]}…\n")

    collection = get_collection()
    ranker = BGEReranker()

    bad = 0
    for qid, e in sorted(exp.items()):
        question, qtype = qmap[qid]
        got = dense_stage(collection, ranker, question)
        want = e["hits"]

        diffs = []
        if len(got) != len(want):
            diffs.append(f"条数 {len(got)} != {len(want)}")
        for i, (g, w) in enumerate(zip(got, want), 1):
            if g["chunk_id"] != w["chunk_id"]:
                diffs.append(f"#{i} chunk_id {g['chunk_id']} != {w['chunk_id']}")
            if round(g.get("rerank_score", 0.0), 4) != w["rerank_score"]:
                diffs.append(f"#{i} score {g['rerank_score']!r} != {w['rerank_score']!r}")

        if diffs:
            bad += 1
            print(f"  FAIL id={qid:<3} [{qtype}] " + "；".join(diffs[:3]))
        else:
            print(f"  ok   id={qid:<3} [{qtype}] {len(got)} 条逐字段一致")

    total = len(exp)
    print(f"\n{'=' * 62}")
    if bad:
        print(f"回归失败：{bad}/{total} 题与卡7 证据不一致 → retrieve() 的改动【不是】等价重构，先别往下走")
    else:
        print(f"回归通过：{total}/{total} 题 逐字段 != 比较全 False → retrieve() 改动对 bge 路径等价")
    print("说明：逐位相同是「改动只换了调用方式、没换数学」的实证，不是 isclose 的近似结论")


if __name__ == "__main__":
    main()