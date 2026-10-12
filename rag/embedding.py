"""嵌入层 | Day9：全项目唯一的 BGE 封装（本文件无 TODO，直接用）。

迁移说明（今天动了哪一行、为什么）：
  实现体从 scripts/common_embedding.py 原样搬到这里，scripts/common_embedding.py
  被改成 3 行兼容壳（re-export）。这样做的原因：项目硬规则是「同一个向量库里所有向量
  必须出自同一个嵌入模型，全项目只允许定义一次嵌入函数」——两份实现迟早漂移。
  01/02/06/07/08 的 import 语句一行没改（它们仍 import common_embedding），
  所以「旧脚本原样保留」的约定仍然成立：脚本不动，只是实现换了个家。
"""
import os

# 兜底：真需要联网时走镜像。用 setdefault 而不是硬赋值——2026-10-07 实测 hf-mirror.com
# 与 huggingface.co 均 connect timeout，硬赋值会把后续所有 HF 下载都钉在一个死地址；
# setdefault 保留外部覆盖能力（当前新模型的下载改走 ModelScope，见 W2 卡2）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_EMBED, DEFAULT_VARIANT, EMBED_MODELS, resolve_variant


# 权重缓存：【模型 path → (tokenizer, model)】。卡6 起同一个模型可以有多档（raw/512/256），
# 按 model_key 缓存会把同一份 0.6B 权重加载三次（白占 ~1.2GB 显存）——按 path 缓存才对齐
# "一份权重、多档维度"的事实。
_LOADED: dict[str, tuple] = {}


def _load(model_key: str) -> tuple:
    """已给：按注册表加载 (tokenizer, model, spec)；约 2.4s / bf16 / 常驻显存。

    左 padding 只对 pooling="last" 开：因果 LM 要取【最后一个真实 token】，左 padding 让
    「最后一个位置」对任何长度都恰好是最后一个真实 token（省掉算 index 这一步 —— 而算 index
    正是最容易错的地方：左 padding 下 mask.sum(dim=1)-1 指向的是【第一个】真实 token，
    长度不等的批里短句会直接落到 padding 上，结果是一批文本拿到同一个向量，且不报错）。
    BERT 系取第 0 个位置的 CLS，所以用默认的右 padding。
    """
    spec = EMBED_MODELS[model_key]
    if spec["path"] not in _LOADED:
        kwargs = {"padding_side": "left"} if spec["pooling"] == "last" else {}
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], **kwargs)
        # transformers 5.x 用 dtype=（torch_dtype 已废弃）；bf16 + GPU + eval
        model = AutoModel.from_pretrained(spec["path"], dtype=torch.bfloat16).to("cuda").eval()
        _LOADED[spec["path"]] = (tokenizer, model)
    tokenizer, model = _LOADED[spec["path"]]
    return tokenizer, model, spec


def _postprocess(pooled: torch.Tensor, dim: int) -> Embeddings:
    """已给：池化后的 (batch, hidden) → 截到 dim 维 → L2 归一化 → CPU list。

    截断是 MRL 的用法（卡6）；dim == 原生维度时它是个空操作。为什么还要显式归一化：
    Chroma 的 cosine 空间内部也会归一化，但向量一旦离开 Chroma（W4 语义缓存要自己算
    相似度）就必须自带单位长度，否则点乘 ≠ 余弦。
    """
    return F.normalize(pooled[:, :dim].float(), p=2, dim=1).cpu().tolist()


class BGEEmbeddingFunction(EmbeddingFunction):
    """bge-small-zh（pooling=cls）：query/doc 同编码，无指令。"""

    def __init__(self, model_key: str = "bge-small-zh", variant: str = DEFAULT_VARIANT):
        self.tokenizer, self.model, self.spec = _load(model_key)
        self.dim = resolve_variant(model_key, variant)["dim"]

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
        return _postprocess(outputs.last_hidden_state[:, 0, :], self.dim)


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """qwen3-0.6b（pooling=last）：【文档侧】编码 —— 不加指令。"""

    def __init__(self, model_key: str = "qwen3-0.6b", variant: str = DEFAULT_VARIANT):
        self.tokenizer, self.model, self.spec = _load(model_key)
        self.dim = resolve_variant(model_key, variant)["dim"]

    def __call__(self, input: Documents) -> Embeddings:
        # 文档侧编码（卡3 已验收）：Chroma 传进来的这一批【块正文】，批长不定。
        # 输出与 input 等长、一一对应、每条 self.dim 维、每条 L2 归一化。
        # 变长批的 padding 在下面 tokenizer 里开。
        tokens = self.tokenizer(
            list(input),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**tokens)
        last_hidden = outputs.last_hidden_state

        attention_mask = tokens["attention_mask"]  # [B, L]
        positions = torch.arange(last_hidden.size(1), device="cuda").unsqueeze(0)
        last_indices = (attention_mask * positions).max(dim=1).values  # [B]

        batch_indices = torch.arange(last_hidden.size(0), device="cuda")
        embedded = last_hidden[batch_indices, last_indices]  # [B, H]

        # 复用 _postprocess：它同时做「截到 self.dim」和「L2 归一化」。
        # 别在这里手写 normalize——那样会跳过截断，卡6 把 dim 改小后维度就对不上了。
        return _postprocess(embedded, self.dim)


# 嵌入函数缓存：加载一次约 2.4s 且常驻显存，绝不能每次调用都重建实例
_EMBEDDER_CACHE: dict[str, EmbeddingFunction] = {}


def get_embedder(model_key: str = DEFAULT_EMBED,
                 variant: str = DEFAULT_VARIANT) -> EmbeddingFunction:
    """已给：按 (model_key, variant) 取嵌入函数实例（带缓存）。

    缓存键必须带上 variant：同一模型的 1024 / 512 / 256 三档【输出维度不同】，
    按 model_key 缓存会让 512 档的调用拿到 1024 维的实例 —— 建 ctx/截断库时即报错。
    权重本身由 _load 按 path 共享，所以多档不额外吃显存，多缓存的只是薄壳。
    """
    cache_key = f"{model_key}::{variant}"
    if cache_key not in _EMBEDDER_CACHE:
        pooling = EMBED_MODELS[model_key]["pooling"]
        if pooling == "cls":
            _EMBEDDER_CACHE[cache_key] = BGEEmbeddingFunction(model_key, variant)
        elif pooling == "last":
            _EMBEDDER_CACHE[cache_key] = Qwen3EmbeddingFunction(model_key, variant)
        else:
            raise ValueError(f"未知 pooling={pooling!r}（model_key={model_key!r}）")
    return _EMBEDDER_CACHE[cache_key]


def encode_query(model_key: str, texts: list[str],
                 variant: str = DEFAULT_VARIANT) -> Embeddings:
    """查询侧统一入口：注册表里有 query_instruction 的模型在这里加指令；没有的等于文档侧编码。

    variant 决定查询向量截到多少维 —— 必须与【目标库建库时】的那一档一致。
    传错档有两种结局：维度不同 → Chroma 当场报错（幸运）；维度相同但截断位置不同 → 不报错，
    距离失去意义（危险），所以查询侧的 variant 一律由调用方显式给，不用默认值兜。

    为什么查询侧不交给 Chroma 自动编码：见 config.EMBED_MODELS 的注释
    （EmbeddingFunction.__call__ 被 add/query 共用，分不出该不该加指令）。
    """
    spec = EMBED_MODELS[model_key]
    fn = get_embedder(model_key, variant)
    if spec["query_instruction"]:
        # 指令拼在【查询】前面，拼完必须再过 fn(...) 真正编码。
        # 用列表推导生成新列表，不要原地改 texts[i]——那会污染调用方手里的原列表。
        texts = [spec["query_instruction"] + t for t in texts]
    return fn(texts)