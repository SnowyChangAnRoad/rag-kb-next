"""
Day 8 | 08_rerank.py
目标：把「向量召回 top10」交给 cross-encoder 精排，取 top3 送生成；
      并先诊断「召回天花板」——rerank 只能重排候选池，救不回没被召回的块。

用法（项目根 rag-kb 下）：
    python scripts/08_rerank.py diag     # 诊断：overlap 那段（p76）在不在 top5/10/20 候选池
    python scripts/08_rerank.py compare  # 对照：同一 query 的「向量 top10」vs「rerank top3」
    python scripts/08_rerank.py test     # 回归：用 Day7 的 10 题跑 rerank 版，和 10/10 对比
    python scripts/08_rerank.py          # 交互问答（召回 top10 → 精排 top3 → 生成）

两阶段检索（通关要能口述）：
    召回  bi-encoder（快 / 求全）：doc 向量可预先算好缓存，可上 ANN → 候选 top10
    精排  cross-encoder（准 / 求准）：query + doc 拼成一个序列做 full attention → top3
    生成  只吃 top3：噪声更少、引用更集中、token 更省

今天最重要的一句话：
    rerank 提高的是 top-k 的「精度」而非「召回率」——候选池里没有的块，rerank 永远排不出来。
    所以先 diag 判断病根：若 p76 不在 top10，那不是排序问题，是 bi-encoder 的召回天花板。

硬边界：非流式（Day12）；单轮问答。
"""
import sys
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common_rag import (
    ROOT, get_client, get_collection, get_llm, retrieve, build_prompt,
    format_sources, show_hits, hit_where, safe_preview,
    SYSTEM_PROMPT, extract_citations, judge_answer,
)

TEST_JSONL = ROOT / "data" / "test_questions.jsonl"

# 9 月已下载模型之外的新模型（Day8 用 HF 镜像下的 bge-reranker-base）
RERANKER_PATH = r"c:\Users\xx\Desktop\python\Agent学习\PyTorch\HuggingFace\models\bge-reranker-base"

RECALL_K = 10            # 第一阶段：向量召回条数（要大于精排数，才有得排）
RERANK_TOP_K = 3         # 第二阶段：精排后送进 prompt 的条数
DIAG_CHUNK_ID = "RAG知识_p076_c001"     # Day6 漏召回的那块（原文含「考虑调整块的重叠大小…」）
DIAG_PAGE = 76
DIAG_K = 20              # 诊断时把候选池挖到 20 条，看清它到底排第几
# 同一意图的三种问法：中英混用（Day6 原问法）/ 纯中文贴近原句 / 关键词堆叠
DIAG_QUERIES = [
    "为什么加 overlap",
    "为什么要在分块时设置块的重叠大小",
    "分块策略调整 重叠大小 语义连接",
]


# ============================================================
# Cross-Encoder：query 与 doc 拼成一条序列，一次前向出一个相关性分数
#   - 与 common_embedding.py 的 BGE 对照：那边是双塔（各自编码，可缓存）；
#     这边是单塔（必须 query+doc 一起进模型，所以无法预先算好）。
# ============================================================
class BGEReranker:
    def __init__(self, model_path: str = RERANKER_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # bge-reranker-base 结构 = BERT + 一层输出 1 个值的分类头
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda").eval()

    # ========================================================
    # TODO 1：给 (query, doc) 对打分，返回 float 列表，约 5 行
    #   输入：question（字符串）、hits（候选块列表）
    #   输出：长度 == len(hits) 的分数列表，每个值在 0~1（越大越相关）
    #   提示：
    #     1) 组对：pairs = [[question, h["text"]], ...]
    #        —— tokenizer 会自动在 query 和 doc 之间插 [SEP]，你不用手拼字符串
    #     2) 编码：self.tokenizer(pairs, padding=True, truncation=True,
    #                             max_length=512, return_tensors="pt").to("cuda")
    #     3) 前向：with torch.no_grad(): logits = self.model(**tokens).logits
    #        —— 形状是 (N, 1)，每行就是这一对的相关性原始分
    #     4) 转概率：torch.sigmoid(...) 把可正可负的 logit 压到 0~1；
    #        记得先 .squeeze(-1) 去掉最后一维，再 .float()（模型是 bf16，转一下更稳）
    #     5) 回 CPU：.cpu().tolist()
    #   自检：分数应是降序趋势且与内容相关，不是全 0.5 一片
    # ========================================================
    def score(self, question: str, hits: list[dict]) -> list[float]:
        pairs = [[question, h["text"]] for h in hits]
        tokens = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**tokens).logits
        scores = torch.sigmoid(logits.squeeze(-1).float()).cpu().tolist()
        return scores

    # ========================================================
    # TODO 2：精排 + 截断，返回 top_k 条，约 5 行
    #   输入：question、hits、top_k
    #   输出：新的 hits 列表（长度 top_k），按精排分数降序
    #   要求：把分数写回每个 hit 的 "rerank_score" 键（便于打印对照），
    #         并保留原 hit 里已有的 "distance"（向量距离）——两个分数一起看才有信息量
    #   提示：scores = self.score(question, hits)
    #         for h, s in zip(hits, scores): h["rerank_score"] = s
    #         再 sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
    #   注意：hits 里是 dict，dict 是可变对象，回写会改到原列表——compare 模式正好靠它对照
    # ========================================================
    def rerank(self, question: str, hits: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        scores = self.score(question, hits)  # 用score()获取分数列表
        for h,s in zip(hits, scores):
            h["rerank_score"] = s
        hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
        return hits

# ------------------------------------------------------------
# 生成（与 Day7 相同的调用，只是喂进去的 hits 变成精排后的 top3）
# ------------------------------------------------------------
def ask_once(llm, question: str, hits: list[dict], verbose: bool = True) -> tuple[str, str]:
    if verbose:
        print("--- 精排后进入 prompt 的资料 ---")
        for rank, h in enumerate(hits, 1):
            score = h.get("rerank_score")
            score_txt = f"s={score:.4f}" if score is not None else ""
            print(f"  【资料{rank}】{score_txt} d={h['distance']:.4f} {h['source']} {hit_where(h)}")
            print(f"      {safe_preview(h['text'])}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    resp = llm.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.7,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = resp.choices[0].message.content
    if verbose:
        print("--- 回答 ---")
        print(answer)
        print("--- 来源列表 ---")
        print(format_sources(hits))
        print(f"Token 用量：输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens}")
    return answer, format_sources(hits)


# ------------------------------------------------------------
# 诊断模式：先弄清「是排序问题还是召回问题」（已给全，直接跑）
#   结论先摆着：rerank 只能重排候选池里的块，捞不回池外的东西。
# ------------------------------------------------------------
def cmd_diag() -> None:
    collection = get_collection(get_client())
    print(f"目标块：{DIAG_CHUNK_ID}（p{DIAG_PAGE}）")
    print("  原文：「…分块策略调整：基于内容类型调整文本块的大小和覆盖率。"
          "考虑调整块的重叠大小以增强文档段之间的语义连接。」")
    print("  它确实存在于 chunks.jsonl / 向量库里，问题只在于「检索能不能捞到它」。")
    print(f"  候选池深度：top{DIAG_K}\n")

    for query in DIAG_QUERIES:
        hits = retrieve(collection, query, DIAG_K)
        rank = next((i for i, h in enumerate(hits, 1) if h["chunk_id"] == DIAG_CHUNK_ID), None)
        n_kw = sum(1 for h in hits if "重叠" in h["text"] or "overlap" in h["text"].lower())
        rank_txt = f"#{rank}" if rank else "未召回"
        print("=" * 70)
        print(f"查询变体：{query}")
        print(f"  p76 在候选池中的名次：{rank_txt}     池内含『重叠/overlap』的块数：{n_kw}")
        print("  该变体的 top3：")
        for i, h in enumerate(hits[:3], 1):
            print(f"    #{i} d={h['distance']:.4f} {h['source']} {hit_where(h)}")
            print(f"        {safe_preview(h['text'], 44)}")

    print("=" * 70)
    print("结论怎么读：")
    print(f"  1) 若所有变体都捞不到 p{DIAG_PAGE} → 是【召回问题】。rerank 无能为力，")
    print("     因为 rerank 只对「已经召回的候选」重排；正解是加 BM25 关键词通道（加餐）或改 query。")
    print(f"  2) 若某个变体能捞到、但排在 {RECALL_K} 名开外 → 调大 RECALL_K，rerank 就有机会。")
    print(f"  3) 若某个变体能排进 top{RECALL_K} 但没进 top{RERANK_TOP_K} → 这正是 rerank 的主场。")
    print("  4) 顺带看清一件事：p76 那块前半截全是 Vectorize 代码（topK/namespace），")
    print("     代码行把「重叠」这句中文的语义稀释了，所以纯中文问法也未必稳。")


# ------------------------------------------------------------
# 对照模式：向量排序 vs 精排排序
# ------------------------------------------------------------
def cmd_compare() -> None:
    question = input(f"对照问题（回车用默认：{DIAG_QUERIES[0]}）> ").strip() or DIAG_QUERIES[0]
    collection = get_collection(get_client())
    ranker = BGEReranker()

    recall = retrieve(collection, question, RECALL_K)
    for rank, h in enumerate(recall, 1):
        h["recall_rank"] = rank     # 记下它原本在向量召回里排第几，精排后好对照

    reranked = ranker.rerank(question, recall, RERANK_TOP_K)

    # ========================================================
    # TODO 3：打印对照表
    #   要求（通关标准就靠它）：
    #     1) 上半区：向量召回 top10，格式 `#名次 d=距离 来源 定位` + 短预览
    #     2) 下半区：精排 top3，格式 `[n] s=精排分 (原向量名次 #x, 距离 d) 来源 定位`
    #     3) 专门标出「被 rerank 救上来的块」：原向量名次 > RERANK_TOP_K 却进了 top3，
    #        用 ← 救上来 之类的标记 —— 这就是你今天要交的「改善实例」
    # ========================================================
    print(f"-------- 向量召回 top{RECALL_K}（按距离升序）--------")
    for h in recall:
        print(f"  #{h['recall_rank']:>2} d={h['distance']:.4f}  {h['source']} {hit_where(h)}")
        print(f"        {safe_preview(h['text'], 44)}")

    print(f"-------- rerank 精排 top{RERANK_TOP_K}（按精排分降序）--------")
    for i, h in enumerate(reranked, 1):
        print(f"  [{i}] s={h['rerank_score']:.4f}  "
              f"(原向量名次 #{h['recall_rank']}, d={h['distance']:.4f})  {h['source']} {hit_where(h)}")
        print(f"        {safe_preview(h['text'], 44)}")
        # 通关要交的「改善实例」：向量里排在 top3 之外，靠 cross-encoder 才挤进来的块
        if h["recall_rank"] > RERANK_TOP_K:
            print(f"        ← 救上来：向量里排 #{h['recall_rank']}，精排后进了 top{RERANK_TOP_K}")
    print("\n对照读法：对比同一来源/页码在两个榜上的名次，就能看出 rerank 到底动了什么。")


# ------------------------------------------------------------
# 回归模式：Day7 的 10 题，改成 rerank 版，看指标是否保持
# ------------------------------------------------------------
def cmd_test() -> None:
    questions = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"读入 {len(questions)} 道测试题（rerank 版：召回 {RECALL_K} → 精排 {RERANK_TOP_K}）")

    collection = get_collection(get_client())
    llm = get_llm()
    ranker = BGEReranker()

    stats: dict[str, list[int]] = {"in": [0, 0], "cross": [0, 0], "out": [0, 0]}
    changed = 0
    for q in questions:
        qid, qtype, question = q["id"], q["type"], q["question"]
        recall = retrieve(collection, question, RECALL_K)
        for rank, h in enumerate(recall, 1):
            h["recall_rank"] = rank
        hits = ranker.rerank(question, recall, RERANK_TOP_K)
        # 精排前后 top3 的 chunk_id 集合是否变化 → 记录 rerank 的实际影响面
        before = {h["chunk_id"] for h in recall[:RERANK_TOP_K]}
        after = {h["chunk_id"] for h in hits}
        if before != after:
            changed += 1

        answer, sources = ask_once(llm, question, hits, verbose=False)
        ok, note = judge_answer(q, answer, hits)
        stats[qtype][0] += int(ok)
        stats[qtype][1] += 1
        mark = "✅" if ok else "❌"
        print("\n" + "=" * 60)
        print(f"[{qid}] ({qtype}) 判定 {mark}  {note}  排序变化={'是' if before != after else '否'}")
        print(f"问题：{safe_preview(question, 50)}")
        print("--- 来源列表（精排后）---")
        print(sources.rstrip() or "(无来源)")
        print(f"--- 回答（前 120 字）---")
        print(safe_preview(answer, 120))

    total_ok = sum(v[0] for v in stats.values())
    total = sum(v[1] for v in stats.values())
    print("\n" + "=" * 60)
    print(f"汇总（rerank 版）：通过 {total_ok}/{total}；top3 排序被 rerank 改动的题数 = {changed}/{total}")
    for t in ("in", "cross", "out"):
        print(f"  {t}: {stats[t][0]}/{stats[t][1]}")
    print("对照：Day7 未 rerank 基线 = in 5/5, cross 3/3, out 2/2（整体 10/10）")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "diag":
        cmd_diag()
        return
    if len(sys.argv) == 2 and sys.argv[1] == "compare":
        cmd_compare()
        return
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        cmd_test()
        return
    if len(sys.argv) == 1:
        collection = get_collection(get_client())
        llm = get_llm()
        ranker = BGEReranker()
        print(f"已加载 {collection.count()} 条知识块 + bge-reranker-base。"
              f"召回 top{RECALL_K} → 精排 top{RERANK_TOP_K}。输入问题开始，q 退出。")
        while True:
            try:
                q = input("\n你的问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in {"q", "quit", "exit"}:
                break
            recall = retrieve(collection, q, RECALL_K)
            for rank, h in enumerate(recall, 1):
                h["recall_rank"] = rank
            hits = ranker.rerank(q, recall, RERANK_TOP_K)
            ask_once(llm, q, hits, verbose=True)
        return
    print("用法：python scripts/08_rerank.py [diag|compare|test]")
    sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================
# 知识三问（Day8 原理层；答案已批改订正，9/19 夜）
# ------------------------------------------------------------
# K6. cross-encoder 与 bi-encoder 的输入形式差在哪？为什么这决定了"能否预缓存"？
#     形式：bi-encoder 把 query 和 doc 分别独立编码成两个向量，相似度靠向量内积算出；
#           cross-encoder 把 query 与 doc 拼成一条序列送进同一个 Transformer，
#           两者 token 之间做 full attention（token 级交互），最后经分类头输出 1 个分数。
#     因果（易答错处）：关键不是"拼在一起"这个动作，而是「同一 doc 的表示随配对的 query 而变」——
#           配不同 query 会得到不同的中间表示，所以 N 篇 doc 无法预先算好存进索引，
#           每次新 query 都要对候选各跑一次前向（O(N) 次/query），也就上不了 ANN。
#           bi-encoder 的 doc 表示与 query 无关，所以能离线预计算 + 建索引 + ANN。
#     一句话面试版：cross-encoder 让 query 与 doc 在同一序列内做 token 级交互，doc 表示随
#           query 而变，因此无法预计算与索引，只能用于小候选集的精排。
# K7. 为什么 rerank 提高的是 precision 而不是 recall？
#     定义：recall 问"候选池里有没有正确答案"；precision@k 问"送进 prompt 的前 k 条里有多少相关"。
#     因果：rerank 的输入是召回阶段定好的固定集合，输出只是这个集合内部的新排列——
#           集合成员不增不减，池里有的还在，池里没有的永远进不来。所以 recall 完全不受影响；
#           变的只是"前 k 条里相关块的比例"，即 precision。
#     实测：问法"为什么加 overlap"时 p76 不在 top20 候选池内 ⇒ 无论怎么重排，输出的 top3
#           必然不含 p76 ⇒ 那个问法的假拒答不可能靠 rerank 修（集合不变 ⇒ 上界被 recall 锁死）。
#     一句话面试版：rerank 只做集合内重排、集合不变，故 recall 由召回阶段锁死，它只能提 precision。
# K8. 精排到 top3 后 token 预算怎么变？为什么 cross 题的判定会变难？
#     预算：k 从 5 降到 3，检索上下文正文量约降到 3/5（对照 07 记录的输入 token 850–1030）。
#     判定：cross 要求"被引用块来自 ≥2 个不同 source"，而 top3 只有 3 个槽位；若被同一 source
#           占满（Day8 的 Q7：p30/p129/p119 全是 RAG知识.pdf），≥2 来源在结构上不可能成立，
#           判定必失败——与模型答得好坏无关。
#           为何容易同源：cross-encoder 逐块判"这块能否回答该问句"，同一语义段相邻块分数接近，
#           会一起挤进前三（与 Day7"同段重复 4 条"同病：单一排序信号导致同源垄断）。
#     补充：rerank 分数是相对量，分数只用于同批内比大小——Q7 的 10 个候选全落在 0.65–0.98，
#           没有低分，其中 p119（讲多轮对话）0.7918 基本是假阳性。
#     → Day9 待决：cross 判定是否改"证据必要性检验"（LLM-as-judge）；cross 题是否单独放宽 top-k。