"""
Day 2 | 02_persistent_chroma.py
目标：PersistentClient 落盘 + 完整 CRUD + where 过滤 + top_k 实验 + 跨进程重启验证。

用法（在项目根目录 rag-kb 下）：
    python scripts/02_persistent_chroma.py build   # 建库：语料向量化并写入磁盘 chroma_db/
    python scripts/02_persistent_chroma.py query   # 查询：重开进程，从磁盘加载后检索
    python scripts/02_persistent_chroma.py crud    # CRUD 演示：get/update/delete/peek/count

关键区别：
    Day1 的 chromadb.Client() 是内存库，进程结束数据消失；
    今天的 PersistentClient(path=...) 把数据落到磁盘，重开进程还在。
"""
import sys
from pathlib import Path
import chromadb
from common_embedding import BGEEmbeddingFunction

# 项目根/chroma_db（scripts 的上一级）。.gitignore 已排除该目录
DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "day2_kb"

# ============================================================
# 语料：20 条，4 个 topic。id 用稳定字符串（建库幂等的关键，见 build 的 upsert）
# ============================================================
CORPUS = [
    # transformer ×6
    ("t01", "transformer", "注意力机制让模型在处理每个词时，动态关注句子中其他相关的词"),
    ("t02", "transformer", "Transformer 架构完全基于自注意力机制，不依赖循环神经网络"),
    ("t03", "transformer", "多头注意力允许模型同时关注不同位置的不同表征子空间"),
    ("t04", "transformer", "Transformer 由编码器和解码器堆叠而成，每层含注意力和前馈网络"),
    ("t05", "transformer", "因果掩码保证解码时每个位置只能看到它之前的 token，防止未来信息泄漏"),
    ("t06", "transformer", "位置编码为注意力机制注入词序信息，RoPE 通过旋转矩阵实现相对位置"),
    # embedding ×6
    ("e01", "embedding", "词嵌入是把离散的词语映射为连续向量的技术"),
    ("e02", "embedding", "余弦相似度衡量两个向量方向的接近程度，常用于语义检索"),
    ("e03", "embedding", "BGE 是 BERT 结构的中文句向量模型，用 CLS 位置输出做句子表示"),
    ("e04", "embedding", "双塔模型分别把查询和文档编码成向量，适合大规模快速召回"),
    ("e05", "embedding", "向量归一化后内积等于余弦相似度，距离取值范围从负一到正一"),
    ("e06", "embedding", "句向量的维度是固定的，BGE-small 输出 512 维稠密向量"),
    # rag ×5
    ("r01", "rag", "RAG 先从外部知识库检索资料，再交给大模型生成回答，以此减少幻觉"),
    ("r02", "rag", "RAG 把非参数化知识存在外部库中，更新知识不需要重新训练模型"),
    ("r03", "rag", "分块是把长文档切成带重叠的小段，每段单独向量化入库"),
    ("r04", "rag", "召回阶段要求快而全，重排阶段用交叉编码器对候选段落精排"),
    ("r05", "rag", "生成提示词应要求模型只依据给定资料回答，资料不足时明确说不知道"),
    # life ×3（作为语义负样本）
    ("l01", "life", "今天天气晴朗，很适合去公园散步晒太阳"),
    ("l02", "life", "周末我打算在家做番茄牛腩，再看一部老电影"),
    ("l03", "life", "楼下咖啡店新出了桂花拿铁，秋天喝正合适"),
]


# ------------------------------------------------------------
# 辅助函数（已给全）：任何子命令都通过它们拿 client / collection，
# 保证所有进程用同一个磁盘路径、同一个嵌入函数、同一个距离空间。
# ------------------------------------------------------------
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(client: chromadb.ClientAPI):
    # get_collection 也要传 embedding_function：query 时需要它给问题向量化
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEEmbeddingFunction(),
    )


def show_results(query: str, results: dict) -> None:
    print(f"\n查询：{query}    (n_results={len(results['ids'][0])})")
    print("-" * 70)
    for rank, (dist, doc, meta, rid) in enumerate(
        zip(results["distances"][0], results["documents"][0],
            results["metadatas"][0], results["ids"][0]),
        start=1,
    ):
        print(f"#{rank}  id={rid}  d={dist:.4f}  [{meta['topic']}]  {doc[:34]}")


# ============================================================
# TODO 1（build）：建库落盘，约 8 行
# ============================================================
def cmd_build() -> None:
    print(f"数据库目录：{DB_PATH}")

    # TODO 1a：用 get_client() 拿持久化客户端；
    #   client.get_or_create_collection(...) 拿集合，三个参数：
    #     name=COLLECTION_NAME
    #     embedding_function=BGEEmbeddingFunction()
    #     metadata={"hnsw:space": "cosine"}
    # 思考：为什么用 get_or_create_collection 而不是 create_collection？
    collection = get_client().get_or_create_collection(
        name = COLLECTION_NAME,
        embedding_function = BGEEmbeddingFunction(),
        metadata = {"hnsw:space": "cosine"}
    )

    ids = [row[0] for row in CORPUS]
    topics = [row[1] for row in CORPUS]
    documents = [row[2] for row in CORPUS]

    # TODO 1b：用 collection.upsert(...) 一次写入全部 20 条
    #   ids=ids, documents=documents,
    #   metadatas=[{"topic": t} for t in topics]
    # 思考：add 遇到已存在的 id 会报错，upsert 遇到已存在的 id 会怎样？
    collection.upsert(
        ids = ids,
        documents = documents,
        metadatas = [{"topic": t} for t in topics]
    )

    # TODO 1c：打印 collection.count()，确认是 20；
    #   再打印一句让用户去资源管理器看 chroma_db/ 目录里生成了什么
    print(f"collection.count()={collection.count()}")
    print(f"去目录查看生成的文件：{DB_PATH}")

# ============================================================
# TODO 2（query）：三个检索实验，每个约 3 行
# 前置：必须先在【另一个进程】跑过 build —— 这正是"重启验证"
# ============================================================
def cmd_query() -> None:
    client = get_client()
    collection = get_collection(client)
    print(f"从磁盘加载成功，当前记录数：{collection.count()}")

    # 实验 1：普通语义检索 top3（和 Day1 一样，确认旧能力在持久库上照常）
    # TODO 2a：query_texts=["Transformer 的编码器是怎么构成的？"], n_results=3
    #          用 show_results 打印；预期 #1 是 t04
    output = collection.query(query_texts=["Transformer 的编码器是怎么构成的？"], n_results=3)
    show_results("Transformer 的编码器是怎么构成的？",output)

    # 实验 2：metadata 过滤 —— 只要 rag 类
    # TODO 2b：query_texts=["模型乱编事实怎么办？"], n_results=3,
    #          where={"topic": {"$eq": "rag"}}
    #          观察：返回的 3 条是否全部 [rag]？对比不加 where 时 t/l 类会不会混入
    output = collection.query(
        query_texts=["模型乱编事实怎么办？"],
        n_results=3,
        where={"topic":{"$eq":"rag"}}
    )
    show_results("模型乱编事实怎么办？",output)
    output = collection.query(
        query_texts=["模型乱编事实怎么办？"],
        n_results=3,
    )
    show_results("模型乱编事实怎么办？",output)


    # 实验 3：多值过滤 + 放大 k —— topic 属于 rag 或 life，取 10 条
    # TODO 2c：query_texts=["秋天适合出门玩"], n_results=10,
    #          where={"topic": {"$in": ["rag", "life"]}}
    # 观察：top3 是不是 life？$in 表示"在给定集合内"；总返回条数受限于命中过滤的数量
    output = collection.query(
        query_texts=["秋天适合出门玩"],
        n_results=10,
        where={"topic": {"$in": ["rag","life"]}}
    )
    show_results("秋天适合出门玩", output)

    # 观察题（运行后把答案写成注释）：
    # O1: 实验2 的 #1 是哪条、distance 多少？如果不加 where，#1 还是它吗？
    #     r01，distance=0.44。如果不加where，#1还是它，但是#2是e04
    # O2: 实验3 一共返回几条？为什么 n_results=10 却不一定返回 10 条？
    #     返回8条
    # O3: build 跑两次以后 count 是 20 还是 40？为什么？
    #     20.不会重复增加数据，幂等性


# ============================================================
# TODO 3（crud）：get / count / peek / update / delete
# 设计：全部操作围绕一条临时记录 "tmp-1"，跑完即删，可重复执行、不污染语料
# ============================================================
def cmd_crud() -> None:
    client = get_client()
    collection = get_collection(client)
    print(f"初始记录数：{collection.count()}")

    # --- 造一条临时记录（已给） ---
    collection.upsert(
        ids=["tmp-1"],
        documents=["这是一条用于 CRUD 演示的临时记录"],
        metadatas=[{"topic": "tmp", "version": 1}],
    )
    print("upsert tmp-1 后 count =", collection.count())

    # TODO 3a：get 按主键精确取（不走向量检索）
    #   item = collection.get(ids=["tmp-1"])
    #   打印 item["documents"][0] 和 item["metadatas"][0]
    item = collection.get(ids=["tmp-1"])
    print("document:", item["documents"][0])
    print("metadata:", item["metadatas"][0])

    # TODO 3b：peek 看库内前 5 条长什么样（已给起点，补打印）
    #   peeked = collection.peek(limit=5)
    #   打印 peeked["ids"]（体会 peek 是"抽样查看"，与语义无关）
    peeked = collection.peek(limit=5)
    print(peeked["ids"])

    # TODO 3c：update 原地改这条记录的原文和 metadata
    #   collection.update(
    #       ids=["tmp-1"],
    #       documents=["临时记录已被更新"],
    #       metadatas=[{"topic": "tmp", "version": 2}],
    #   )
    #   再 get 一次，打印 document 和 version，确认变更生效
    #   思考：改了 document 以后，它的向量是谁、在什么时候重新算的？
    collection.update(
        ids = ["tmp-1"],
        documents = ["临时记录已被更新"],
        metadatas = [{"topic":"tmp","version":2}]
    )
    getion = collection.get(ids=["tmp-1"])
    print(getion["documents"][0],getion["metadatas"][0]["version"])

    # TODO 3d：删除 + 验证
    #   collection.delete(ids=["tmp-1"])
    #   打印 collection.count()（应回到 20）
    #   再 collection.get(ids=["tmp-1"])，观察被删 id 返回的 documents 是什么（空列表）
    collection.delete(ids=["tmp-1"])
    print(collection.count())
    print(collection.get(ids=["tmp-1"]))

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "query", "crud"}:
        print("用法：python scripts/02_persistent_chroma.py [build|query|crud]")
        sys.exit(1)
    {"build": cmd_build, "query": cmd_query, "crud": cmd_crud}[sys.argv[1]]()
    # sys.argv 表示命令行参数列表。sys.argv[0]为运行文件路径。sys.argv[1]即第一个参数。
    # 命令行输入 python scripts/02_persistent_chroma.py build 即运行cmd_build函数
    # 等价的 if/elif 版（更好懂） if sys.argv[ 1 ] == "build" :


if __name__ == "__main__":
    main()
