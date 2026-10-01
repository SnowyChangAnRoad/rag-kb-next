"""
Day 4 | 04_load_web.py
目标：解析 data/raw/ 下的 HTML 文件，去导航栏/脚本噪声提取正文，写入 data/processed/web_documents.jsonl。

用法（在项目根 rag-kb 下）：
    python scripts/04_load_web.py inspect    # 体检：每份标题/URL/噪声比/首段预览
    python scripts/04_load_web.py extract   # 正式提取：写 web_documents.jsonl（覆盖写，幂等）

硬边界：删噪声标签 + 全文提取；不做 CSS 选择器精准定位正文容器。
"""
import re
import sys
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_JSONL = PROCESSED_DIR / "web_documents.jsonl"

# Ctrl+S 存的 HTML 里 <script> 占绝大部分体积；这些标签整棵子树删掉
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe"]

# 已验证的 URL 映射（canonical 提取失败时的兜底）
URL_FALLBACK = {
    "RAG 技术概览.html": "https://zhuanlan.zhihu.com/p/678893732",
    "Transformer--Attention Is All You Need.html": "https://zhuanlan.zhihu.com/p/689083488",
}


# ------------------------------------------------------------
# 辅助函数（已给全）
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """空白归一化（从 03_load_pdf.py 复制；Day9 重构进 rag/ 包时合并）。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def list_htmls() -> list[Path]:
    return sorted(RAW_DIR.glob("*.html"))


def safe_preview(text: str, n: int = 200) -> str:
    t = clean_text(text).replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


# ============================================================
# TODO 1：extract_url —— 从 HTML 自身挖 URL，约 6 行
#   优先级：canonical <link> → og:url <meta> → 兜底映射表
#   提示：find("link", rel="canonical") 返回 Tag 或 None
#         Tag.get("href") 拿属性值；拿不到就进下一层
# ============================================================
def extract_url(soup: BeautifulSoup, filename: str) -> str:
    # TODO 1a：找 <link rel="canonical">，取 href
    link = soup.find("link",rel="canonical")
    if link and link.get("href"):
        return link["href"]

    # TODO 1b：找 <meta property="og:url">，取 content
    meta = soup.find("meta",property="og:url")
    if meta and meta.get("content"):
        return meta["content"]

    # 兜底（已给）
    return URL_FALLBACK.get(filename, "")


# ============================================================
# TODO 2：extract_title —— 优先 <h1>，其次 <title>，约 5 行
#   提示：find("h1") 返回 Tag 或 None
#         Tag.get_text(strip=True) 拿纯文本
#         soup.title 也可能是 None
# ============================================================
def extract_title(soup: BeautifulSoup) -> str:
    # TODO 2a：先试 <h1>
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(strip=True)
        if t:
            return t

    # TODO 2b：<h1> 没有就试 <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return ""


# ============================================================
# TODO 3（inspect）：体检，约 15 行
#   对每份 HTML 打印：文件名、标题、URL、
#   原始文本长度（decompose 前）vs 清洗后长度（decompose 后+clean_text）、
#   噪声占比%、首段预览
# ============================================================
def cmd_inspect() -> None:
    htmls = list_htmls()
    if not htmls:
        print(f"未在 {RAW_DIR} 找到 HTML 文件")
        return

    for html_path in htmls:
        raw_bytes = html_path.read_bytes()
        soup = BeautifulSoup(raw_bytes, "html.parser")

        title = extract_title(soup)
        url = extract_url(soup, html_path.name)

        # TODO 3a：decompose 前先量一下原始文本长度
        raw_text = soup.get_text()
        raw_len = len(raw_text)

        # TODO 3b：decompose 所有噪声标签
        for tag in NOISE_TAGS:
            for t in soup.find_all(tag):
                t.decompose()

        # TODO 3c：decompose 后取正文并清洗
        cleaned = clean_text(soup.get_text())
        cleaned_len = len(cleaned)
        noise_pct = 100 * (1 - cleaned_len / raw_len) if raw_len else 0

        print(f"\n{'=' * 60}")
        print(f"文件：{html_path.name}")
        print(f"标题：{title}")
        print(f"URL：{url}")
        print(f"原始文本：{raw_len} 字 → 清洗后：{cleaned_len} 字（噪声 {noise_pct:.1f}%）")
        print(f"首段预览：{safe_preview(cleaned)}")



# ============================================================
# TODO 4（extract）：写 jsonl，约 15 行
#   每篇一条 JSON：{"doc_id", "source", "url", "title", "text", "char_count"}
#   疑问：每篇一条JSON是否合适？
#   doc_id 规则："{stem}_web"，如 RAG技术概览_web
# ============================================================
def cmd_extract() -> None:
    htmls = list_htmls()
    if not htmls:
        print(f"未在 {RAW_DIR} 找到 HTML 文件")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for html_path in htmls:
            soup = BeautifulSoup(html_path.read_bytes(), "html.parser")
            title = extract_title(soup)
            url = extract_url(soup, html_path.name)

            # TODO 4a：decompose 噪声标签（和 inspect 里一样）
            for tag in NOISE_TAGS:
                for t in soup.find_all(tag):
                    t.decompose()

            # TODO 4b：取正文 + 清洗
            text = clean_text(soup.get_text())
            if not text:
                print(f"  跳过空正文：{html_path.name}")
                continue

            # TODO 4c：构造记录并写一行
            record = {
                "doc_id": f"{html_path.stem}_web",
                "source": html_path.name,
                "url": url,
                "title": title,
                "text": text,
                "char_count": len(text)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += 1
            print(f"  写入：{html_path.name} → {len(text)} 字")

    print(f"\n完成：写入 {written} 条")
    print(f"输出：{OUTPUT_JSONL}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"inspect", "extract"}:
        print("用法：python scripts/04_load_web.py [inspect|extract]")
        sys.exit(1)
    {"inspect": cmd_inspect, "extract": cmd_extract}[sys.argv[1]]()


if __name__ == "__main__":
    main()
