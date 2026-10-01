"""切分层 | Day12：文本清洗 + 递归切块（从 scripts/03_load_pdf.py 与 05_split.py 抽出）。

为什么要抽出来（Day12 卡 3 的前置动作）：
    上传 PDF 要做的「逐页解析 → 清洗 → 剔目录/残页 → 递归切块」与 Day3/Day5 脚本
    逐字相同。若在 api/main.py 里重写一遍，项目里就有两套切块口径：
    上传的块 chunk_id 形态、overlap 长度、目录页判定都会慢慢漂移，
    而 Day14 的指标表要求「库内所有块可比」—— 口径一漂，指标就没有意义了。
    这就是 Day9 抽 rag/embedding.py 的同一条理由：真代码只允许有一份。

依赖方向：splitter → （仅标准库），不依赖 config / retriever / llm。
    纯函数、无 IO（除了 pdf_pages_to_docs 读 PDF 文件），所以可以单独测。

历史脚本的关系：
    scripts/05_split.py 已改为从本模块 import（保留 CLI 与统计打印，核心逻辑不再自带）；
    scripts/03_load_pdf.py 的 extract 是历史脚本、保留不动，其逐页提取逻辑与
    pdf_pages_to_docs 等价（那边多了一步写 documents.jsonl 落盘）。

硬边界：纯文本处理；不做 OCR、不做版面分析、不做语义分块。
"""
import re

from pathlib import Path

import pdfplumber

# 分块层级：段落 → 行 → 句 → 空格
SEPARATOR_LEVELS = [
    ["\n\n"],
    ["\n"],
    ["。", "！", "？"],
    [" "],
]

MIN_CHARS = 100      # 短于该字数的记录剔除（Day3 发现的 p101 目录残页仅 22 字）
TOC_RATIO = 0.3      # 点线行占比超该阈值判为目录页（Day3 发现的 p005）
OVERLAP_PCT = 0.1    # overlap = chunk_size * 10%（字符数）


# ------------------------------------------------------------
# 清洗
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """空白归一化：全角空格转半角、行内多空格压单空格、3+空行压双换行、首尾 strip。

    （Day3 搬来）注意这里【不】过滤字符集：PDF 抽出来的中文里混着的英文/数字是有效内容，
    真正要防的是排版噪声（全角空格、连续空行），不是字符种类。
    """
    if not text:
        return ""
    text = text.replace("\u3000", " ")          # 全角空格 → 半角
    text = re.sub(r"[ \t]+", " ", text)         # 行内连续空白 → 单空格
    text = re.sub(r"[ \t]+\n", "\n", text)      # 行尾空白
    text = re.sub(r"\n[ \t]+", "\n", text)      # 行首空白
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3 个及以上换行 → 双换行
    return text.strip()


# ------------------------------------------------------------
# 页级过滤（剔目录页 / 残页）
# ------------------------------------------------------------
def is_toc_page(text: str) -> bool:
    """点线行占比 ≥ TOC_RATIO → 判为目录页。

    规则：一行内出现 ≥5 次 " ." 视为点线行（如 "第1章 引言 . . . . . 1"，
    clean_text 已把多空格压成单空格，所以点线是 " . . ." 模式）。
    必须是「空格+点」：裸点会误杀英文句号/URL 多的长行（Day5 实测教训）。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    dot_lines = 0
    for ln in lines:
        if ln.count(" .") >= 5:
            dot_lines += 1
    return dot_lines / len(lines) >= TOC_RATIO


def filter_docs(docs: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """剔除目录页/残页。返回 (保留列表, [(被剔记录, 原因), ...])。

    为什么过滤要发生在切块【之前】：目录页切出来的块是「第1章…第2章…」的碎片，
    它跟任何问题都只有弱相似度，却会在精排时偶尔挤进候选池 —— 属于纯噪声源，
    在源头剔掉比在下游调 top_k 便宜得多（Day2「先过滤后 top-k」的同一模式）。
    """
    kept, dropped = [], []
    for doc in docs:
        text = doc["text"]
        if len(text) < MIN_CHARS:
            dropped.append((doc, f"短页：{len(text)} 字 < {MIN_CHARS}"))
        elif is_toc_page(text):
            dropped.append((doc, "目录页：点线行占比超阈"))
        else:
            kept.append(doc)
    return kept, dropped


# ------------------------------------------------------------
# 递归切块
# ------------------------------------------------------------
def recursive_split(text: str, level_idx: int, chunk_size: int) -> list[str]:
    """递归切分（不带 overlap）。

    1. 整段 ≤ chunk_size → 原样返回（保持段落完整，递归的意义所在）
    2. 超长 → 用当前层分隔符切开，贪心合并成接近 chunk_size 的块
    3. 单个片段仍超长 → 降到下一层递归切
    4. 所有层用尽 → 按 chunk_size 硬切（最后手段）

    分隔符接回前一段末尾（seg += parts[i+1]）是为了不丢标点 ——
    切掉句号会让「句末」这个语义边界消失，且块首尾对比时看起来像截断。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if level_idx >= len(SEPARATOR_LEVELS):          # 所有层用尽 → 硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pattern = "(" + "|".join(re.escape(s) for s in SEPARATOR_LEVELS[level_idx]) + ")"
    parts = re.split(pattern, text)                 # 切成 [片段, 分隔符, 片段, 分隔符, ...]
    segments = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if i + 1 < len(parts):
            seg += parts[i + 1]                     # 分隔符粘回前一段尾
        if seg.strip():
            segments.append(seg)

    chunks, buf = [], ""                            # buf = 当前正在填充的块
    for seg in segments:
        if len(seg) > chunk_size:                   # 单段仍超长 → 降层递归
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(recursive_split(seg, level_idx + 1, chunk_size))
        elif buf and len(buf) + len(seg) > chunk_size:
            chunks.append(buf)
            buf = seg
        else:
            buf += seg
    if buf:
        chunks.append(buf)
    return chunks


def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """同一条记录内，每块开头拼上前一块尾部 overlap 个字符。

    只在同一条记录（同一页/同一篇）内部加，绝不跨记录 ——
    跨文档的内容拼进块里会污染语义和来源（Day5 的硬边界）。

    代价（Day8 才看清）：overlap 让相邻块共享约 30 字，embedding 与精排分都极接近，
    于是同源几块会一起被推进 top3，把跨文档证据挤出去 ——
    这正是后来要 dedupe_adjacent + select_diverse 的原因。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append(prev[-overlap:] + cur)
    return out


def build_chunks(docs: list[dict], chunk_size: int) -> list[dict]:
    """流水线：递归切分 → 同记录内加 overlap → 构造带 metadata 的块记录。

    chunk_id = f"{doc_id}_c{i:03d}"，如 RAG知识_p076_c001。
    这个命名不是装饰：retriever.parse_chunk_id 靠它拆出 (doc_id, seq) 来判「相邻冗余」，
    去重规则完全建立在这个格式上 —— 改格式会让去重静默失效。
    """
    overlap = int(chunk_size * OVERLAP_PCT)
    all_chunks = []
    for doc in docs:
        raw = recursive_split(doc["text"], 0, chunk_size)
        chunks = add_overlap(raw, overlap)
        for i, ck in enumerate(chunks, 1):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_c{i:03d}",
                "doc_id": doc["doc_id"],      # 追溯到源记录（页/篇）
                "source": doc["source"],
                "page": doc.get("page"),      # 网页无页码 → None
                "url": doc.get("url"),        # PDF 无 URL → None
                "title": doc.get("title"),
                "text": ck,
                "char_count": len(ck),
            })
    return all_chunks


# ------------------------------------------------------------
# PDF → 页级 docs（Day12 卡 3：上传入口复用）
# ------------------------------------------------------------
def pdf_pages_to_docs(pdf_path) -> list[dict]:
    """逐页解析文本型 PDF，清洗后返回页级 docs（每页一条）。

    与 scripts/03_load_pdf.py 的 extract 等价，区别只是不落 documents.jsonl。

    两个必须守住的点：
      ① page = page.page_number，不是列表下标 —— 下标从 0 起，页码从 1 起，
         错位 1 会让所有「第x页」引用整体偏移（Day3 的坑）。
      ② doc_id 用文件 stem：上传同名文件时 doc_id 相同 → chunk_id 相同 → upsert 覆盖，
         天然幂等（这正是下面 build_chunks 能直接喂 build_index 的前提）。
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = clean_text(page.extract_text())
            if not text:
                continue                              # 空页跳过（扫描件、纯图页）
            docs.append({
                "doc_id": f"{Path(pdf_path).stem}_p{page.page_number:03d}",
                "source": Path(pdf_path).name,
                "page": page.page_number,
                "text": text,
                "char_count": len(text),
            })
    return docs


def pdf_to_chunks(pdf_path, chunk_size: int) -> tuple[list[dict], int, int]:
    """上传入口用的一站式：PDF → 过滤 → 切块。

    返回 (chunks, 页数, 被剔页数) —— 页数与剔除数要给用户看：
    上传 20 页却只入库 3 块时，接口必须能解释「另外 17 页去哪了」（目录页/残页/扫描件）。
    静默丢数据是上传接口最忌讳的行为（Day11 /health 吞异常的同一个道理）。
    """
    docs = pdf_pages_to_docs(pdf_path)
    kept, dropped = filter_docs(docs)
    return build_chunks(kept, chunk_size), len(docs), len(dropped)
