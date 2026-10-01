"""嵌入层 | Day9：全项目唯一的 BGE 封装（本文件无 TODO，直接用）。

迁移说明（今天动了哪一行、为什么）：
  实现体从 scripts/common_embedding.py 原样搬到这里，scripts/common_embedding.py
  被改成 3 行兼容壳（re-export）。这样做的原因：项目硬规则是「同一个向量库里所有向量
  必须出自同一个嵌入模型，全项目只允许定义一次嵌入函数」——两份实现迟早漂移。
  01/02/06/07/08 的 import 语句一行没改（它们仍 import common_embedding），
  所以「旧脚本原样保留」的约定仍然成立：脚本不动，只是实现换了个家。
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 兜底：真需要联网时走镜像

import torch
import torch.nn.functional as F
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer

from .config import EMBED_MODEL_PATH


class BGEEmbeddingFunction(EmbeddingFunction):
    """Chroma 嵌入函数：add/query 时被 Chroma 回调，句子列表 -> L2 归一化向量列表。"""

    def __init__(self, model_path: str = EMBED_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # transformers 5.x 用 dtype=（torch_dtype 已废弃）；bf16 + GPU + eval
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda").eval()

    def __call__(self, input: Documents) -> Embeddings:
        tokens = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=512,          # bge-small-zh 上限，超长会被静默截断（Day5 结论）
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**tokens)
        # BGE 用 CLS pooling：第 0 个 token 的隐藏状态作为句向量
        cls_emb = outputs.last_hidden_state[:, 0, :].float()
        # L2 归一化后内积 == 余弦相似度；Chroma 只收 CPU 上的 Python list
        return F.normalize(cls_emb, p=2, dim=1).cpu().tolist()