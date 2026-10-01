"""
Day 3 | 03_load_pdf.py
目标：逐页解析 data/raw/ 下的文本型 PDF，清洗后按页写入 data/processed/documents.jsonl。

用法（在项目根 rag-kb 下）：
    python scripts/03_load_pdf.py inspect    # 只体检：每份页数、空页数、首页预览，确认无乱码
    python scripts/03_load_pdf.py extract    # 正式提取：逐页写 documents.jsonl（覆盖写，幂等）

硬边界：只处理文本型 PDF；不做 OCR、不做版面分析、不做表格结构化。
"""
import re
import sys
import json
from pathlib import Path

import pdfplumber

# scripts 的上一级是项目根；data/ 下 raw 放原始 PDF，processed 放产出
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_JSONL = PROCESSED_DIR / "documents.jsonl"


# ------------------------------------------------------------
# 辅助函数（已给全）
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """空白归一化：全角空格转半角、行内多空格压单空格、3+空行压双换行、首尾 strip。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ")          # 全角空格 → 半角
    text = re.sub(r"[ \t]+", " ", text)         # 行内连续空白 → 单空格
    text = re.sub(r"[ \t]+\n", "\n", text)      # 行尾空白
    text = re.sub(r"\n[ \t]+", "\n", text)      # 行首空白
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3 个及以上换行 → 双换行
    return text.strip()


def list_pdfs() -> list[Path]:
    return sorted(RAW_DIR.glob("*.pdf"))


def safe_preview(text: str, n: int = 200) -> str:
    """控制台只打印短预览，避免长文本里的特殊 Unicode 触发 cp936 编码错误。"""
    t = clean_text(text).replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


# ============================================================
# TODO 1（inspect）：体检，约 15 行
#   要求对每份 PDF 打印：文件名、总页数、空页数、
#   扫一遍PDF，不写入数据
#   第一页 extract_text 的前 200 字预览（用 safe_preview）。
# ============================================================
def cmd_inspect() -> None:
    pdfs = list_pdfs()
    if not pdfs:
        print(f"未在 {RAW_DIR} 找到 PDF")
        return

    for pdf_path in pdfs:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            empty = 0
            first_preview = ""
            for page in pdf.pages:
                # TODO 1a：text = page.extract_text()
                #   若 text 为空（None 或 strip 后为空），empty += 1；continue
                #   若 first_preview 仍为空，把 safe_preview(text) 赋给它
                # 思考：为什么不能直接用 page.extract_text().strip()？
                text = page.extract_text()
                if text is None or not text.strip():
                    empty += 1
                    continue
                if first_preview == "":
                    first_preview = safe_preview(text)

            print(f"\n{'=' * 60}")
            print(f"文件：{pdf_path.name}")
            # TODO 1b：打印 总页数 total / 空页数 empty / 首段预览 first_preview
            #   格式参考：
            #   print(f"总页数：{total}    空页数：{empty}")
            #   print(f"首段预览：{first_preview}")
            print(f"总页数：{total}    空页数：{empty}")
            print(f"首段预览：{first_preview}")


# ============================================================
# TODO 2（extract）：逐页提取并写 jsonl，约 15 行
#   每页一条 JSON：{"doc_id", "source", "page", "text", "char_count"}
#   doc_id 规则："{文件名stem}_p{页码:03d}"，如 RAG知识_p001
# ============================================================
def cmd_extract() -> None:
    pdfs = list_pdfs()
    if not pdfs:
        print(f"未在 {RAW_DIR} 找到 PDF")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    written = 0
    skipped = 0

    # "w" 覆盖写：同 Day2 upsert 思路，重跑不重复
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for pdf_path in pdfs:
            stem = pdf_path.stem           # .stem属性：路径最后一部分的文件名，去掉后缀
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    total_pages += 1
                    raw = page.extract_text()
                    text = clean_text(raw)
                    if not text:
                        skipped += 1
                        continue

                    # TODO 2a：构造记录 dict
                    #   doc_id  = f"{stem}_p{page.page_number:03d}"
                    #   source  = pdf_path.name
                    #   page    = page.page_number   ← 必须用这个，不能用列表下标
                    #   text    = text
                    #   char_count = len(text)
                    record = {"doc_id":f"{stem}_p{page.page_number:03d}",
                              "source":pdf_path.name,
                              "page":page.page_number,
                              "text":text,
                              "char_count":len(text)}

                    # TODO 2b：写一行 jsonl
                    #   f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    #   ensure_ascii=False 必须加，否则中文变 \uXXXX 转义，没法肉眼抽查
                    f.write(json.dumps(record,ensure_ascii=False)+"\n")

                    written += 1

    print(f"\n完成：扫描 {total_pages} 页，写入 {written} 条，跳过空页 {skipped} 条")
    print(f"输出：{OUTPUT_JSONL}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"inspect", "extract"}:
        print("用法：python scripts/03_load_pdf.py [inspect|extract]")
        sys.exit(1)
    {"inspect": cmd_inspect, "extract": cmd_extract}[sys.argv[1]]()


if __name__ == "__main__":
    main()
