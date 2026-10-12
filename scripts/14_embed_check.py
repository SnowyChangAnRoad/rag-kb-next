"""验收 | W2 卡3：Qwen3-Embedding 接入自检（验收标准 ①–⑤ + 建 qwen3_kb）。

断言（全可量化，任一不过就 FAIL）
  ① 维度/归一化   encode_query 返回 (n, 1024)，每条 L2 norm == 1.0
  ② 指令生效      cos(带指令的 query, 相关块) > cos(带指令的 query, 无关块)
  ③ 变长批不塌缩  一次编码 3 条【长度差异大】的文本，两两 cos 不得全为 1.0
  ④ 确定性        同一 query 连编两次，逐元素 != 比较必须全 False
  ⑤ 建库          qwen3_kb 条数 == chunks.jsonl 行数；且能用 1024 维查询向量命中

为什么 ③ 要单列一条
    左 padding 下用 mask.sum(dim=1)-1 取 index：批次内长度不等时，短句的 index 会落到
    padding 上 → 一批【不同】文本拿到【同一个】向量，而且不报错。
    只校验「index 处的 mask == 1」抓不到它 —— 那对任何真实 token 都成立。
    这条断言是照今天真实踩到的坑写的（cos(q,rag) == cos(q,weather) == 0.0299）。

用法（项目根 rag-kb-next 下）
    & "C:\\Users\\xx\\Desktop\\python\\.venv\\Scripts\\python.exe" scripts/14_embed_check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import EMBED_MODELS, resolve_variant  # noqa: E402
from rag.embedding import encode_query, get_embedder  # noqa: E402
from rag.ingest import build_index, load_chunks  # noqa: E402
from rag.retriever import get_collection  # noqa: E402

KEY = "qwen3-0.6b"
INSTR_KEY = "query_instruction"


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<18} {detail}")
    return ok


def main() -> None:
    spec = EMBED_MODELS[KEY]
    dim = resolve_variant(KEY)["dim"]
    print(f"卡3 验收 | model_key={KEY} | dim={dim} | pooling={spec['pooling']} "
          f"| mrl={spec['mrl']} | variants={spec['variants']}\n")

    try:
        # ① 维度 / 归一化
        enc = encode_query(KEY, ["什么是 RAG", "如何配置向量库"])
    except NotImplementedError as e:
        print(f"TODO 未填（{e}）→ 先实现 TODO-1 / TODO-2 再跑本脚本。")
        return

    ok1 = report("① 维度", all(len(v) == dim for v in enc), f"shape=({len(enc)}, {len(enc[0])})")
    norms = [sum(x * x for x in v) ** 0.5 for v in enc]
    # 阈值 1e-5 而非 1e-9：向量是 float32（eps≈1.19e-7），1024 维求和会累积到 ~1e-6。
    # 真没归一化的向量范数是 0.3 / 5.0 量级，1e-5 足够区分。
    ok1 &= report("① 归一化", all(abs(n - 1.0) < 1e-5 for n in norms),
                  f"norms={[round(n, 9) for n in norms]}")

    # ② 指令生效（只加查询侧才有这个对比度）
    q, rel, irr = "什么是 RAG", "RAG 是检索增强生成，先召回再生成", "今天天气不错，适合出门散步"
    qv, relv, irrv = encode_query(KEY, [q]), encode_query(KEY, [rel]), encode_query(KEY, [irr])
    c_rel, c_irr = cos(qv[0], relv[0]), cos(qv[0], irrv[0])
    ok2 = report("② 指令生效", c_rel > c_irr, f"cos(相关)={c_rel:.4f} > cos(无关)={c_irr:.4f}")

    # ③ 变长批不塌缩（文档侧，正是 Chroma 的批量调用形态）
    docs = get_embedder(KEY)(["RAG", "RAG 是检索增强生成，先召回再生成后精排", "中" * 200])
    pairs = [(1, 2), (1, 3), (2, 3)]
    coss = [round(cos(docs[i - 1], docs[j - 1]), 4) for i, j in pairs]
    ok3 = report("③ 变长批不塌缩", not all(c == 1.0 for c in coss), f"两两 cos={coss}")

    # ④ 确定性
    again = encode_query(KEY, [q])[0]
    same = all(a == b for a, b in zip(qv[0], again))
    ok4 = report("④ 确定性", same, "同一 query 连编两次逐元素 != 全 False")

    # ⑤ 建库 + 维度对齐（1024 维查询向量能命中，说明库维度与编码维度一致）
    chunks = load_chunks()
    col = get_collection(model_key=KEY)
    n = build_index(col, chunks)
    ok5 = report("⑤ 建库", n == len(chunks),
                 f"{resolve_variant(KEY)['collection']} count={n} vs chunks={len(chunks)}")
    hit = col.query(query_embeddings=enc[:1], n_results=1)["ids"][0]
    ok5 &= report("⑤ 维度对齐", len(hit) == 1, f"查询命中 {hit}")

    allok = all([ok1, ok2, ok3, ok4, ok5])
    print(f"\n{'=' * 58}\n{'卡3 验收通过' if allok else '卡3 验收未过'}："
          f"①{dim}维 {int(ok1)}/2 ②{int(ok2)} ③{int(ok3)} ④{int(ok4)} ⑤{int(ok5)}/2")
    print("⑥ 回归（bge 路径逐位不变）另跑 scripts/15_regress_dense.py")


if __name__ == "__main__":
    main()