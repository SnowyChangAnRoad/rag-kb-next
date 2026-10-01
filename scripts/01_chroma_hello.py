"""
Day 1 | 01_chroma_hello.py
目标：用本地 bge-small-zh 作为 Chroma 的嵌入函数，完成 add → query 最小闭环。
背景：Chroma 默认嵌入函数是英文 ONNX 模型（国内下载极慢），故提前启用 Day2 的本地 BGE 方案。
前置：9 月的 06_embedding.py（CLS pooling + L2 归一化）。
"""
import chromadb

# Day2 重构：嵌入函数抽到公共模块，全项目只此一份（同一向量库必须用同一嵌入模型）
# 同目录脚本可直接 import：运行 python scripts/01_chroma_hello.py 时，
# Python 自动把脚本所在目录 scripts/ 加入 sys.path，不依赖运行时的工作目录。
from common_embedding import BGEEmbeddingFunction


def show_results(query: str, results: dict) -> None:
    """格式化打印一次检索结果（已给全，直接用）"""
    print(f"\n查询：{query}")
    print("-" * 70)
    for rank, (dist, doc, meta) in enumerate(
        zip(results["distances"][0], results["documents"][0],
            results["metadatas"][0]),
        start=1,
    ):
        print(f"#{rank}  distance={dist:.4f}  [{meta['topic']}]  {doc}")


def main() -> None:
    # 内存客户端：进程结束数据即消失（Day2 换 PersistentClient 落盘）
    client = chromadb.Client()

    # --------------------------------------------------------
    # TODO 2：创建集合 + 写入文档（约 12 行）
    # --------------------------------------------------------
    # TODO 2a：create_collection，三个参数：
    #   name="day1_hello"
    #   embedding_function=BGEEmbeddingFunction()   ← 不用默认英文模型
    #   metadata={"hnsw:space": "cosine"}           ← 距离空间选余弦
    collection = client.create_collection(
        name="day1_hello",
        embedding_function=BGEEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    documents = [
        "注意力机制让模型在处理每个词时，动态关注句子中其他相关的词",
        "Transformer 架构完全基于自注意力机制，不依赖循环神经网络",
        "词嵌入是把离散的词语映射为连续向量的技术",
        "余弦相似度衡量两个向量方向的接近程度，常用于语义检索",
        "RAG 先从外部知识库检索资料，再交给大模型生成回答，以此减少幻觉",
        "今天天气晴朗，很适合去公园散步晒太阳",
    ]
    topics = ["transformer", "transformer", "embedding", "embedding", "rag", "life"]

    # TODO 2b：collection.add(...)
    #   documents=documents
    #   ids=[f"d{i}" for i in range(len(documents))]   ← id 必须是唯一字符串
    #   metadatas=[{"topic": t} for t in topics]       ← Day2 where 过滤要用
    collection.add(
        documents=documents,
        ids=[f"d{i}" for i in range(len(documents))],
        metadatas=[{"topic": t} for t in topics],
    )

    # --------------------------------------------------------
    # TODO 3：检索（约 6 行）
    # 对下面两个 query 各调一次 collection.query(query_texts=[q], n_results=3)
    # 并用 show_results(q, results) 打印
    # --------------------------------------------------------
    queries = [
        "自注意力是什么？",
        "怎么判断两个向量在语义上接近？",
    ]
    # 提示：for q in queries: results = ...; show_results(q, results)
    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        show_results(q, results)

    # --------------------------------------------------------
    # 观察题（运行后把答案写在下面，不用写代码）：
    # O1: #1 distance = 0.4368 / 0.2395；life 是 metadata 的 topic 标签（散步句），
    #     n_results=6 时垫底排第 6，distance=0.7318
    # O2: distance = 1 - 余弦相似度，理论范围 [0,2]；distance=0 表示两向量方向完全一致
    # O3: 排序不变。L2 归一化后 ||a-b||^2 = 2(1-cos)，l2 距离恰好是 cosine distance 的 2 倍
    #     （实测 0.8736 vs 0.4368），只差常数倍，故排序相同
    # --------------------------------------------------------


if __name__ == "__main__":
    main()
