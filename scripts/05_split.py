"""
Day 5 | 05_split.py
目标：把 documents.jsonl（131 条 PDF 页）+ web_documents.jsonl（2 条网页）
     递归分块成 chunks.jsonl，每块带 chunk_id / 来源 / 页码等 metadata。

用法（在项目根 rag-kb 下）：
    python scripts/05_split.py inspect      # 三组参数(300/500/800)统计对比，不写文件
    python scripts/05_split.py split 300    # 用选定参数正式写 chunks.jsonl（覆盖写，幂等）

Day12 改动（重要）：核心逻辑已迁到 rag/splitter.py，本文件只保留 CLI 与统计打印。
    为什么迁：Day12 的上传接口要做「PDF → 清洗 → 剔目录 → 递归切块」，
    与这里逐字相同。若在 API 里重写一份，项目里就有两套切块口径 ——
    上传的块与现有 531 块就不可比，Day14 的指标表随之失效。
    这与 Day9 把嵌入实现收进 rag/embedding.py、把本文件的老搭档
    common_embedding.py 改成兼容壳是同一个动作：真代码只允许有一份。

    搬走的函数（现在 from rag.splitter import）：
        clean_text / is_toc_page / filter_docs / recursive_split / add_overlap / build_chunks
    原 Day5 的 TODO 1–4 代码一字未改，只是换了住处，含注释一起搬走了。

设计决策（通关会考）：
1. 按"条"切：PDF 一页=一条、网页一篇=一条。代价：书的内容跨页处语义被硬切断，
   属已知局限，Day6 检索效果不佳再优化（先跑通再优化）。
2. 分块层级 \n\n → \n → 。！？ → 空格：官方默认层级为英文设计（空格=词边界），
   中文 PDF 文本里空格多为排版噪声，语义价值低，故句末标点排在空格前。
3. overlap 为字符数（chunk_size 的 10%），加块后块长可能略超 chunk_size，正常现象。
4. 不引入 LangChain，只手写递归切分学原理。

硬边界：纯文本处理（标准库）；不做语义分块、不做按文档结构分块。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包，运行时把项目根塞进 sys.path 才能 import rag.*（同 common_embedding.py）
sys.path.insert(0, str(ROOT))

from rag.splitter import (          # noqa: E402
    OVERLAP_PCT, build_chunks, filter_docs,
)

PROCESSED_DIR = ROOT / "data" / "processed"
PDF_JSONL = PROCESSED_DIR / "documents.jsonl"
WEB_JSONL = PROCESSED_DIR / "web_documents.jsonl"
OUTPUT_JSONL = PROCESSED_DIR / "chunks.jsonl"

SIZES = (300, 500, 800)


# ------------------------------------------------------------
# 辅助函数（已给全）
# ------------------------------------------------------------
def safe_preview(text: str, n: int = 40) -> str:
    """控制台只打印短预览，避免特殊 Unicode 触发 cp936 编码错误（Day3 教训）。"""
    return text.replace("\n", " ")[:n]


def load_documents() -> list[dict]:
    """合并读入 PDF 页 + 网页两类记录（字段有差异：网页多 url/title，无 page）。"""
    docs = []
    for path in (PDF_JSONL, WEB_JSONL):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def print_stats(chunks: list[dict], label: str) -> None:
    """打印一组块的基本统计 + 按块长分布的简易直方图（桶宽 100 字）。"""
    if not chunks:
        print(f"\n--- {label}：0 块 ---")
        return
    lengths = [c["char_count"] for c in chunks]
    avg = sum(lengths) / len(lengths)
    print(f"\n--- {label}：{len(chunks)} 块 | 平均 {avg:.0f} 字 "
          f"| 最短 {min(lengths)} | 最长 {max(lengths)} ---")

    buckets: dict[int, int] = {}
    for n in lengths:
        buckets[n // 100] = buckets.get(n // 100, 0) + 1
    max_bar = max(buckets.values())
    for b in sorted(buckets):
        bar = "#" * round(buckets[b] / max_bar * 40)
        print(f"  {b * 100:4d}-{b * 100 + 99:4d} | {bar} {buckets[b]}")


# ------------------------------------------------------------
# 子命令
# ------------------------------------------------------------
def cmd_inspect() -> None:
    """三组参数对比（300 / 500 / 800），不写文件。"""
    docs = load_documents()
    kept, dropped = filter_docs(docs)

    print(f"读入 {len(docs)} 条，剔除 {len(dropped)} 条：")
    for doc, reason in dropped:
        print(f"  ✗ {doc['doc_id']}（{len(doc['text'])} 字）— {reason}")

    for size in SIZES:
        chunks = build_chunks(kept, size)
        print_stats(chunks, f"chunk_size={size} overlap={int(size * OVERLAP_PCT)}")
        if len(chunks) >= 2:
            print(f"  [overlap 抽样] 前块尾：{safe_preview(chunks[0]['text'][-40:])}")
            print(f"  [overlap 抽样] 后块头：{safe_preview(chunks[1]['text'][:40])}")


def cmd_split(size: int) -> None:
    """正式写 chunks.jsonl（覆盖写，幂等）。"""
    docs = load_documents()
    kept, dropped = filter_docs(docs)
    chunks = build_chunks(kept, size)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ck in chunks:
            f.write(json.dumps(ck, ensure_ascii=False) + "\n")
    print(f"输出总块数：{len(chunks)}")
    print(f"输出路径：{OUTPUT_JSONL}")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "inspect":
        cmd_inspect()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "split" and sys.argv[2] in {"300", "500", "800"}:
        cmd_split(int(sys.argv[2]))
        return
    print("用法：python scripts/05_split.py [inspect | split 300|500|800]")
    sys.exit(1)


if __name__ == "__main__":
    main()
