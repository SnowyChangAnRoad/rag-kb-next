"""入库层 | Day9：chunks.jsonl → Chroma 集合（等价于 Day6/07 的 build 子命令，只是搬进包里）。

依赖方向：ingest → embedding → config，不反向依赖 retriever/pipeline。
"""
import json
import os

from .config import BATCH_SIZE, CHUNKS_JSONL
from .embedding import BGEEmbeddingFunction   # noqa: F401  （连库时由调用方建 collection 用）


def load_chunks(path=CHUNKS_JSONL) -> list[dict]:
    """已给：读 chunks.jsonl，一行一个 dict（Day14 交付时 932 条）。"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def save_chunks(chunks: list[dict], source: str | None = None, path=CHUNKS_JSONL) -> int:
    """把新块并入真相源 chunks.jsonl（先删同 source 旧行、再追加），返回总行数。

    【Day13 验收发现后新加】为什么必须有它：
        /files 上传原先只写向量库、不写 chunks.jsonl —— 上传的块成了「重建追不回」的数据。
        Day9 定的口径是「chunks.jsonl 是真相源、向量库可重建」，而上传入口绕过了真相源：
        一旦调 POST /ingest 重建（或 HNSW 损坏后按老办法重建），这些块会【静默消失】。
        Day13 验收时上传的 Redis全套学习笔记.pdf（401 块）就是这种状态 ——
        库里有 932 块，真相源只有 531 块，重建即掉到 531 且不报任何错。

    为什么按 source 删旧行，而不是直接追加：
        重传同名 PDF 在向量库里是 upsert 覆盖（chunk_id 相同），若 jsonl 只追加，
        真相源里就会留下「同名 source 的两代块」—— 重建时旧块集体复活，且无法分辨版本。
        删同 source 旧行 + 追加新行，才与向量库的语义对齐。

    原子性：写同目录临时文件再 os.replace 覆盖。
        真相源被写坏（写一半）比丢一次上传严重得多 —— 它是唯一的恢复依据。
        os.replace 在同盘内是原子操作，中途失败最多留下一个 .tmp 垃圾文件，
        不会出现「读得进一行、读不进下一行」的半截真相源。
    """
    existing = load_chunks(path)
    if source:
        existing = [c for c in existing if c.get("source") != source]
    rows = existing + list(chunks)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return len(rows)


def clean_meta(chunk: dict) -> dict:
    """已给：Chroma 的 metadata 只接受 str/int/float/bool。

    chunks.jsonl 里 PDF 块的 url 是 None、网页块的 page 是 None，
    带 None 直接喂 Chroma 会抛错（Day6 踩过），所以这里删掉空值键，并剥掉正文 text。
    """
    return {k: v for k, v in chunk.items() if v is not None and k != "text"}


def build_index(collection, chunks: list[dict]) -> int:
    """分批 upsert 写入库，返回写入后的总数。

    - 分 BATCH_SIZE 条一批：一次性提交 932 条会给内存与底层协议无谓压力（Day6 定为 64）
    - 用 upsert 而不是 add：重跑不报错（Day2 教训：add 遇已存在 id 会抛异常）
    - metadatas 必须先过 clean_meta：删掉 None 键（Chroma 只收 str/int/float/bool）
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: min(len(chunks),i+BATCH_SIZE)]
        collection.upsert(ids=[c["chunk_id"] for c in batch],
                          documents=[c["text"] for c in batch],
                          metadatas=[clean_meta(c) for c in batch])
    return collection.count()