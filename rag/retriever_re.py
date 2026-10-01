""" 检索层 - 手动复现
    召回 -> 精排 -> 去重(相邻去重、+同源配额)
"""
import re
import chromadb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    DB_PATH, RECALL_K, RERANKER_MODEL_PATH, COLLECTION_NAME,
)
from .embedding import BGEEmbeddingFunction

# ---------------------- 工具函数 ----------------------
def get_client() -> chromadb.ClientAPI:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client

def get_collection(client=None):
    client = client if client else get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEEmbeddingFunction,
        metadata={"hnsw:space": "cosine"},
    )

def phrase_chunk_id(chunk_id : str) -> tuple[str,int]:
    m = re.match(r"^(.*)_c(\d+)$", chunk_id)
    return (m.group(1), int(m.group(2))) if m else (chunk_id, -1)

# ---------------------- 召回 ----------------------
def retriever(collection, question: str, k: int = RECALL_K) -> list[dict]:
    output = collection.query(query_texts=[question], n_results=k)
    hits = []
    for dist, doc, meta, cid in zip(
        output["distances"][0], output["documents"][0],
        output["metadatas"][0], output["ids"][0]
    ):
        hits.append({"chunk_id": cid, "distance": dist, "text": doc, **meta})
    return hits

# ---------------------- 精排 ----------------------
class BGEReanker:
    def __init__(self, model_path = RERANKER_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda").eval()
    def score(self, question: str, hits: list[dict]) -> list[float]:
        pairs = [[question, h["text"]] for h in hits]
        tokens = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad:
            logits = self.model(**tokens).logits
        return torch.sigmoid(logits.squeeze(-1).float()).cpu().tolist()
    def rerank(self, question: str, hits: list[dict]) -> list[dict]:
        scores = self.score(question, hits)
        for h, s in hits, scores:
            h["rerank_score"] = s
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

# ---------------------- 相邻去重 ----------------------
def depute_adjacent(hits: list[dict]) -> list[dict]:
    kept, keys = [], []
    for h in hits:
        docs, seq = phrase_chunk_id(h["ids"])
        if(any(docs == d and abs(seq - s)<=1) for d, s in keys):
            continue
        kept.append(h)
        keys.append((docs,seq))
    return kept

# ---------------------- 同源配额+截断 ----------------------
def select_diverse(hits: list[dict], k: int, per_source_cap: int) -> list[dict]:
    picked = []
    overflow = []
    used = {}

    for h in hits:
        if len(picked) >= k :
            break
        src = h["source"]
        if used.get(src,0) >= per_source_cap :
            overflow.append(h)
        else:
            picked.append(h)

    if len(picked) < k:
        for h in overflow:
            if len(picked) >= k:
                break
            picked.append(h)

    return picked