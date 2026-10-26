"""评测层 | Day10：30 题 × 3 组检索/生成策略，出「优化前 → 优化后」对照表。

Day10 的口径切换（一次做完，之后所有对比都在新口径下可比）：
    1) 温度 0.7 → 0
    2) 10 题 → 30 题
    3) 新增「引用覆盖率」指标 cite_at_k —— 见下面 TODO-R2/TODO-R3 与 summarize 的注释

指标：
    答案正确率   = correct 题数 / 总题数（LLM 裁判 1，不看引用）
    引用正确率   = 有效引用条数 / 模型实际写出的引用条数（旧口径，保留对照）
    引用覆盖率   = 被有效引用的编号数 / top_k（新口径，分母受实验控制）
    平均检索延迟 = recall_ms + rerank_ms（不含生成）
    平均生成用量 = prompt_tokens / completion_tokens，以及按 config 单价折算的成本

用法（在项目根 rag-kb 下，且需要关闭沙箱以访问 chroma 文件）：
    python -m rag.eval            端到端：30 题 × 各组，每题 1 次生成 + 2 次裁判（花 API 钱）
    python -m rag.eval retrieval  只跑检索层：召回→精排→去重→配额，逐题比 top-k 构成
                                  （零 API 成本、结果完全可复现 → 判「配额有没有用」看这个）
    python -m rag.eval final      只跑交付口径一组（Day14），落盘 config.EVAL_FINAL_JSON
结果同时打印到控制台并落盘 config.EVAL_JSON（Day9 的 eval_day9.json 留在磁盘作旧口径存档）。
"""

import json
import re
import time

from itertools import combinations

from .config import (CROSS_TOP_K, CROSS_TYPES, EVAL_JSON, RECALL_K, RERANK_TOP_K,
                     SOURCE_CAP, TEST_JSONL, EVAL_FINAL_JSON)
from .embedding import BGEEmbeddingFunction
from .judge import judge_citations, judge_correct
from .pipeline import RAGPipeline
from .llm import safe_preview
from .retriever import (BGEReranker, dedupe_adjacent, get_client, get_collection,
                        parse_chunk_id, retrieve, select_diverse)

# 实验分组：一次只动一个变量，才能把指标差异归因到某一处改动（Day9 的教训）。
# 卡 2/3 的结论：A→B 在**端到端**上不可信（11 题里 8 题两组送进 prompt 的资料逐字相同、
#   指标却仍会变），所以配额的验收改在检索层做（python -m rag.eval retrieval：零误伤 + 重复槽下降）。
# 卡 4 要看的是 B→C：唯一变量是 system prompt（拒答门槛 v1 → v2）。A/B 保留作同轮锚点。
GROUPS = [
    {"name": "A_基线(无配额)", "dedupe": True, "top_k": 3, "quota_cap": None, "prompt": "v1"},
    {"name": "B_+配额cap2", "dedupe": True, "top_k": 3, "quota_cap": SOURCE_CAP, "prompt": "v1"},
    {"name": "C_+拒答门槛v2", "dedupe": True, "top_k": 3, "quota_cap": SOURCE_CAP, "prompt": "v2"},
]


def load_questions() -> list[dict]:
    questions = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


# ------------------------------------------------------------
# 引用覆盖率——为什么需要它
# ------------------------------------------------------------
# 旧口径「有效引用 / 模型写出的引用条数」的分母是模型的自由度：Day9 实测同温度下
# Q2 写了 7 条、Q4 只写 1 条 → 组间对比时分母同步漂移，指标差异无法归因。
# 新口径把分母钉成 top_k，语义随之变成「给它的 top-k 资料，有几条真的被当成证据用上了」。
# 它不依赖裁判 2 的 total（那是「引用条数」），只依赖「哪些编号被判无效」。
CITE_RE = re.compile(r"\[(\d+)\]")


# 从回答中抽出引用编号 去重+排序
def extract_cited_ids(answer: str) -> list[int]:
    output = [int(n) for n in CITE_RE.findall(answer)]
    return sorted(set(output))


# 筛选有效引用
def count_valid_cited(cited_ids: list[int], invalid, top_k: int) -> list[int]:
    # 参数：cited_ids 来自 extract_cited_ids()； invalid 来自 judge_citations(...)["invalid"]；top_k 是本组实际条数。
    invalid = [int(n) for n in invalid]
    invalid = sorted(set(invalid))
    output = []
    for i in cited_ids:
        if (i not in invalid) and i <= top_k :
            output.append(i)
    return output


# 跑一组：每题 = 1 次生成 + 2 次裁判。prompt 是 system prompt 版本（卡 4 的变量）
def run_group(pipeline: RAGPipeline, questions: list[dict], dedupe: bool, top_k,
              quota_cap=None, prompt: str = "v1") -> list[dict]:
    records = []
    for q in questions:
        out = pipeline.answer(q["question"], qtype=q["type"], dedupe=dedupe,
                              top_k=top_k, quota_cap=quota_cap, prompt_version=prompt)
        jc = judge_correct(q["question"], out["hits"], out["answer"])
        jcit = judge_citations(q["question"], out["hits"], out["answer"])
        cited = extract_cited_ids(out["answer"])                        # 模型写了哪些编号
        valid_ids = count_valid_cited(cited, jcit["invalid"], out["top_k"])
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "prompt": out["prompt_version"],
            "top_k": out["top_k"], "n_recall": out["n_recall"], "n_dedup": out["n_dedup"],
            "sources": [f"{h['source']}#{h.get('page') or h.get('url')}" for h in out["hits"]],
            "retrieve_ms": out["retrieve_ms"], "gen_ms": out["gen_ms"],
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "cost": out["cost"],
            "correct": jc["correct"], "correct_reason": jc["reason"],
            "cite_total": jcit["total"], "cite_valid": jcit["valid"],
            "cite_invalid": jcit["invalid"], "cite_missing": jcit["missing"],
            "cited_ids": cited, "valid_cited_ids": valid_ids,
            "answer": out["answer"],
        }
        records.append(rec)
        mark = "✅" if rec["correct"] else "❌"
        cite_txt = f"{rec['cite_valid']}/{rec['cite_total']}" if rec["cite_total"] else "无引用"
        cover_txt = f"{len(valid_ids)}/{rec['top_k']}"
        print(f"    [{rec['id']:>2}] {mark} 引用 {cite_txt:<6} 覆盖 {cover_txt:<5} "
              f"答案 {len(rec['answer']):>4}字 检索 {rec['retrieve_ms']:>5.0f}ms 顶{rec['top_k']}条 | "
              f"{safe_preview(rec['question'], 30)}")
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    cite_total = sum(r["cite_total"] for r in records)
    cite_valid = sum(r["cite_valid"] for r in records)
    n_zero_cite = sum(1 for r in records if r["cite_total"] == 0)
    # 新口径直接由明细汇总：有效编号总数 / 总槽位数（pooled，比「比值的平均」更稳）
    slot_valid = sum(len(r["valid_cited_ids"]) for r in records)
    slot_total = sum(r["top_k"] for r in records)

    def rate(rs: list[dict]) -> float:
        return sum(r["correct"] for r in rs) / len(rs) if rs else 0.0

    by_type = {}
    for t in ("in", "cross", "partial", "out"):
        rs = [r for r in records if r["type"] == t]
        by_type[t] = {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                      "rate": rate(rs)}
    return {
        "n": n,
        "answer_acc": rate(records),
        "cite_acc": (cite_valid / cite_total) if cite_total else 0.0,
        "cite_total": cite_total, "cite_valid": cite_valid,
        "cite_at_k": (slot_valid / slot_total) if slot_total else 0.0,
        "slot_valid": slot_valid, "slot_total": slot_total,
        "zero_cite_questions": n_zero_cite,
        "avg_retrieve_ms": sum(r["retrieve_ms"] for r in records) / n,
        "avg_gen_ms": sum(r["gen_ms"] for r in records) / n,
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in records) / n,
        "avg_completion_tokens": sum(r["completion_tokens"] for r in records) / n,
        "avg_cost": sum(r["cost"] for r in records) / n,
        "by_type": by_type,
    }


# ------------------------------------------------------------
# 检索层判决（Day10 新增）—— 零 API 成本、结果可复现
# ------------------------------------------------------------
# 为什么必须单开这一层：卡 2 的 A/B 端到端对比里，11 题中有 8 题两组送进 prompt 的资料
# 完全相同（prompt_tokens 逐题相等；Q6 连 chunk_id 和顺序都逐条相等），指标却仍有差异 ——
# 那部分差异来自生成侧/裁判侧的抖动，不是实验变量造成的。判「配额有没有用」先把 LLM 摘掉：
# 只跑 召回 → 精排 → 去重 → 配额，比 top-k 的构成。同一份数据、同一份代码，重跑永远一致。

EXPECT_PATTERNS = [
    ("RAG知识", "RAG知识.pdf"),
    ("技术概览", "RAG 技术概览.html"),
    ("Transformer", "Transformer--Attention Is All You Need.html"),
]

# 召回 + 精排 + 去重
def retrieve_only(collection, ranker, question: str, top_k: int, quota_cap: int | None) -> list[dict]:
    hits = retrieve(collection, question, RECALL_K)
    hits = ranker.rerank(question, hits=hits)
    hits = dedupe_adjacent(hits)
    if quota_cap:
        hits = select_diverse(hits, top_k, quota_cap)
    else:
        hits = hits[:top_k]
    return hits

# 计算 top-k 中「白占槽位」的重复
def redundancy_stats(hits: list[dict], ef) -> dict:
    """TODO-R2：数出 top-k 里「白占槽位」的重复 —— 顺手给出 MMR 的立项依据。

    三档重复，粒度从细到粗，正好对应三种机制：
        page_dup  同一页占了 ≥2 条 → dedupe_adjacent 的辖区
                  （页 = parse_chunk_id(chunk_id) 的 doc_id，如 "RAG知识_p016"）
        file_dup  同一文件（source 字段）占了 ≥2 条 → select_diverse 的辖区
        max_cos   top-k 两两余弦相似度的最大值 → 前两者都碰不到的区间
                  「不同页 / 不同来源但语义重复」。**这是 MMR 唯一能治的场景**：
                  普遍接近 1 才有立项理由，普遍 0.7 上下则 MMR 是在跟噪声较劲。

    计数口径 = 「重复占用的槽位数」，即每个分组的（出现次数 - 1）之和：
        3 条同源 → 2；1 条 → 0。这样 A 组 Q9（Transformer ×3）记 2、B 组（×2）记 1。
    余弦怎么算：ef(hits 的 text 列表) 返回的向量已 L2 归一化 → 余弦 = 点积
        （numpy 的 @ 或 sum(a*b) 都行；top-k 最多 5 条，两两组合用 itertools.combinations）
    返回 {"page_dup": int, "file_dup": int, "max_cos": float}（不足 2 条时 max_cos 记 0.0）
    """
    page_dup = len(hits) - len(set([parse_chunk_id(h["chunk_id"])[0] for h in hits]))
    file_dup = len(hits) - len(set([h["source"] for h in hits]))
    vecs = ef([h["text"] for h in hits])  # 一次批量算完，别在循环里逐条调（模型前向的开销在启动，不在条数）
    m = 0.0
    for va, vb in combinations(vecs, 2):  # 从 5 条里取全部两两组合
        m = max(m, sum(x * y for x, y in zip(va, vb)))
    return {"page_dup": page_dup, "file_dup": file_dup, "max_cos": m}


def expect_hit(hits: list[dict], expect: str) -> tuple[int, int]:
    """TODO-R3：用 test_questions.jsonl 的 expect 字段给 top-k 做「召回代理指标」。

    expect 有三种粒度，都要能吃：
        库内题   "RAG知识.pdf 第16页"                     → 期望到【页】：top-k 里要有该文件该页
        跨文档题 "需同时引用 RAG知识.pdf 与 RAG 技术概览"  → 期望到【文件集合】：两个 source 都要在
        库外题   "库外，应拒答"                            → 没有期望，返回 (0, 0)，汇总时跳过

    做法：
        1) 用 EXPECT_PATTERNS 扫 expect（子串匹配）得到期望的文件名列表；一个都没匹配到 → (0, 0)
        2) 页码用正则 r"第(\\d+)页" 抠；只在「期望文件恰好一个」时要求页码
           （跨文档题的页码只出现在括号注释里，不该当硬要求）
        3) 命中判定：hits 里存在 source 相同的块；要求页码时还须 page 相同
    返回 (命中数, 期望数)。它只看检索结果 —— 不受生成、也不受裁判抖动影响。
    """
    files = [name for kw, name in EXPECT_PATTERNS if kw in expect]
    if not files:
        return 0, 0  # "库外，应拒答" 会自然落到这里，不用特判
    m = re.search(r"第(\d+)页", expect)
    need_page = int(m.group(1)) if (m and len(files) == 1) else None
    hit = sum(1 for f in files
              if any(h["source"] == f and (need_page is None or h.get("page") == need_page) for h in hits))
    return hit, len(files)


# 检索层判决主流程：逐题打印 top-k 构成 + 汇总表，不调用任何 LLM。
def retrieval_main() -> None:
    questions = load_questions()
    collection = get_collection(get_client())
    ranker = BGEReranker()
    ef = BGEEmbeddingFunction()      # 模型只加载一次，循环里复用（Day9 依赖注入的同一习惯）
    print(f"检索层判决：{len(questions)} 题 × {len(GROUPS)} 组"
          f"（只跑 召回/精排/去重/配额，不调 LLM；同一份数据重跑结果一致）")

    rows = {}
    for g in GROUPS:
        top_k_txt = "按题型" if not g["top_k"] else str(g["top_k"])
        quota_txt = f"cap{g['quota_cap']}" if g.get("quota_cap") else "无配额"
        print(f"\n{'=' * 92}\n组：{g['name']}（dedupe={g['dedupe']}, top_k={top_k_txt}, {quota_txt}）")
        recs = []
        for q in questions:
            k = g["top_k"] or (CROSS_TOP_K if q["type"] in CROSS_TYPES else RERANK_TOP_K)
            hits = retrieve_only(collection, ranker, q["question"], k, g.get("quota_cap"))
            st = redundancy_stats(hits, ef)
            hit, tot = expect_hit(hits, q["expect"])
            recs.append({"id": q["id"], "type": q["type"], "top_k": k, **st,
                         "expect_hit": hit, "expect_total": tot})
            srcs = " ".join(f"{h['source']}#{h.get('page') or 'web'}" for h in hits)
            print(f"    [{q['id']:>2}] {q['type']:<5} 期望 {hit}/{tot}  "
                  f"同页重复 {st['page_dup']} 同源重复 {st['file_dup']} "
                  f"maxcos {st['max_cos']:.3f} | {srcs}")
        rows[g["name"]] = recs

    # ---------------- 汇总 ----------------
    print(f"\n{'=' * 92}\n检索层汇总（分母固定，不含生成/裁判）\n")
    print(f"{'组':<20}{'同源重复槽/题':<15}{'同页重复槽/题':<15}{'平均maxcos':<12}{'期望来源命中':<14}")
    print("-" * 92)
    for name, recs in rows.items():
        n = len(recs)
        print(f"{name:<20}{sum(r['file_dup'] for r in recs) / n:>11.2f}    "
              f"{sum(r['page_dup'] for r in recs) / n:>11.2f}    "
              f"{sum(r['max_cos'] for r in recs) / n:>8.3f}    "
              f"{sum(r['expect_hit'] for r in recs)}/{sum(r['expect_total'] for r in recs):<10}")

    print("\nMMR 立项判据（看 C 组：配额已治不了的语义重复还剩多少）")
    base = rows[list(rows)[-1]]
    for r in sorted(base, key=lambda x: x["max_cos"], reverse=True)[:3]:
        verdict = "有活干（≥0.9，配额/去重都碰不到）" if r["max_cos"] >= 0.9 else "重复度不高，MMR 收益有限"
        print(f"  [{r['id']:>2}] {r['type']:<5} maxcos {r['max_cos']:.3f}  → {verdict}")

    print("\n怎么读这张表（通关会问）：")
    print("  1) 同源重复槽数：B 组应明显低于 A 组，但**不该为 0** —— cap2 是允许同源占 2 条的；")
    print("  2) 期望来源命中：这是 top-k 的 recall 代理（可复现）。配额如果砍掉了高价值来源，")
    print("     它就会掉 —— 这就是「配额会不会误伤」的检验；")
    print("  3) maxcos：接近 1 说明 top-k 内部语义近乎重复，那是配额和去重都覆盖不到的区间。")


def main() -> None:
    questions = load_questions()
    print(f"读入 {len(questions)} 道测试题；分 {len(GROUPS)} 组跑（每组 = {len(questions)} 生成 + "
          f"{len(questions) * 2} 裁判）")
    pipeline = RAGPipeline(get_collection(get_client()))

    result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}
    for g in GROUPS:
        quota_txt = f"quota_cap={g['quota_cap']}" if g.get("quota_cap") else "无配额"
        print(f"\n{'=' * 72}\n组：{g['name']}（dedupe={g['dedupe']}, "
              f"top_k={g['top_k'] or '按题型'}, {quota_txt}, prompt={g['prompt']}）")
        records = run_group(pipeline, questions, g["dedupe"], g["top_k"],
                            g.get("quota_cap"), g["prompt"])
        summary = summarize(records)
        result["groups"][g["name"]] = {"summary": summary, "records": records}

    # ---------------- 汇总表 ----------------
    print(f"\n{'=' * 72}\nDay10 对照表（{len(questions)} 题 × {len(GROUPS)} 组，温度 0）\n")
    header = (f"{'组':<20}{'答案正确率':<12}{'引用正确率':<12}{'引用覆盖率':<12}{'零引用题':<10}"
              f"{'检索ms':<10}{'生成ms':<10}{'输入tok':<10}{'输出tok':<10}{'单次成本元':<10}")
    print(header)
    print("-" * 122)
    for name, payload in result["groups"].items():
        s = payload["summary"]
        cite_k = f"{s['slot_valid']}/{s['slot_total']}={s['cite_at_k'] * 100:.0f}%"
        print(f"{name:<20}{s['answer_acc'] * 100:>7.1f}%    "
              f"{s['cite_acc'] * 100:>7.1f}%    {cite_k:<12}{s['zero_cite_questions']:<10}"
              f"{s['avg_retrieve_ms']:>7.0f}   {s['avg_gen_ms']:>7.0f}   "
              f"{s['avg_prompt_tokens']:>7.0f}   {s['avg_completion_tokens']:>7.0f}   "
              f"{s['avg_cost']:>8.4f}")

    print("\n分题型答案正确率：")
    for name, payload in result["groups"].items():
        bt = payload["summary"]["by_type"]
        detail = "  ".join(f"{t} {bt[t]['correct']}/{bt[t]['n']}"
                            for t in ("in", "cross", "partial", "out") if bt[t]['n'])
        print(f"  {name:<20}{detail}")

    # ---------------- 结论怎么读 ----------------
    print("\n怎么读这张表（通关问题会问）：")
    print("  1) 引用正确率（旧口径，分母由模型决定）与引用覆盖率（新口径，分母 = top_k）")
    print("     要放一起看：若某组旧口径低、新口径也低，说明模型没用好资料；")
    print("     若旧口径低但新口径正常，说明是「全量标注」把旧指标拉低了，检索其实没问题。")
    print("  2) 只看总正确率会掩盖互相抵消：必须看下面的分题型数据 ——")
    print("     同源配额预期在 cross 题上救回另一侧证据，在 in 题（答案就在同一份 PDF）可能变差。")
    print("  3) 每组之间只差一个变量（A 基线 / B +配额 / C +拒答门槛），")
    print("     否则指标差异无法归因 —— 这是 Day9 指标抖动教训的直接应用。")
    print("  4) 检索延迟应远小于生成延迟（召回+精排是本地 GPU，生成要过网络）——")
    print("     这是下一阶段埋点延迟分解的基线参照。")

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n明细已落盘：{EVAL_JSON}")


# -------------------Final--------------------------
# 交付口径：必须与 api/main.py 的 ChatRequest 默认值逐项对应，
# 否则「README 里的指标」和「服务实际行为」是两码事
FINAL_GROUP = {"name": "final_交付口径", "dedupe": True, "top_k": RERANK_TOP_K,
               "quota_cap": SOURCE_CAP, "prompt": "v2"}

def final_main() -> None:
    """TODO-1：只跑最终交付口径一组，落盘 EVAL_FINAL_JSON。"""
    questions = load_questions()
    collection = get_collection(get_client())
    pipeline = RAGPipeline(collection)
    records = run_group(pipeline, questions, FINAL_GROUP["dedupe"], FINAL_GROUP["top_k"],
                        FINAL_GROUP["quota_cap"], FINAL_GROUP["prompt"])

    summary = summarize(records)
    # 打印要能让「验收」一眼判过没过：单位统一成 %、与验收线 80% 并排。
    # 顺带把 README 指标表要用的数字（引用覆盖率 / 延迟 / token / 成本 / 分题型）一起打出来 ——
    # summarize 已经算好了，不打白不打，否则填 README 时还得回头翻 json。
    print(f"\n最终指标（锚定库 {collection.name}：{collection.count()} 块）")
    print(f"  答案正确率 {summary['answer_acc'] * 100:.1f}%  "
          f"{'✅' if summary['answer_acc'] >= 0.8 else '❌'} ≥80%（总验收 #2）")
    print(f"  引用正确率 {summary['cite_acc'] * 100:.1f}%  "
          f"{'✅' if summary['cite_acc'] >= 0.8 else '❌'} ≥80%（总验收 #2）")
    print(f"  引用覆盖率 {summary['slot_valid']}/{summary['slot_total']}="
          f"{summary['cite_at_k'] * 100:.1f}%（分母固定 = top_k，Day10 新口径）")
    print(f"  零引用题 {summary['zero_cite_questions']} 题")
    print(f"  平均检索 {summary['avg_retrieve_ms']:.0f}ms / 平均生成 {summary['avg_gen_ms']:.0f}ms")
    print(f"  平均输入 {summary['avg_prompt_tokens']:.0f} tok / 输出 "
          f"{summary['avg_completion_tokens']:.0f} tok / 单次 ¥{summary['avg_cost']:.4f}")
    bt = summary["by_type"]
    print("  分题型正确率：" + "  ".join(
        f"{t} {bt[t]['correct']}/{bt[t]['n']}" for t in ("in", "cross", "partial", "out")))

    result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "library": {"collection": collection.name, "count": collection.count()},
              "group": FINAL_GROUP, "summary": summary, "records": records}
    with open(EVAL_FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"明细已落盘：{EVAL_FINAL_JSON}")
    
    # 读 EVAL_JSON（Day10 存档）把 C 组同名指标并排打出来。
    # 【批改】原写法 `with open(EVAL_JSON, "r", encoding=False):` 有两个坑：
    #   ① encoding 只能给编码名字符串或 None，给 False 直接报错（Windows 上不写则回落 GBK，
    #      读 ensure_ascii=False 写出的中文 json 会乱码/报 UnicodeDecodeError，必须显式 utf-8）；
    #   ② 漏了 `as f` → 下面的 json.load(f) 用的是上面那个【已关闭】的 f
    #      （ValueError: I/O operation on closed file）。看着像在读 EVAL_JSON，其实没打开过。
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        group_c_summary = json.load(f)["groups"]["C_+拒答门槛v2"]["summary"]
    print("\n对照 Day10 C 组（+拒答门槛v2，与本次同为 quota_cap=2 + prompt v2）：")
    print(f"  答案正确率 {group_c_summary['answer_acc'] * 100:.1f}% / "
          f"引用正确率 {group_c_summary['cite_acc'] * 100:.1f}% / "
          f"引用覆盖率 {group_c_summary['cite_at_k'] * 100:.1f}%")
    print(f"  ⚠️ Day10 锚在 531 块库，本次锚在 {collection.count()} 块库 —— "
          f"数字差异含【库版本差异】，不可直接归因到策略")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "retrieval":
        retrieval_main()          # 零成本、可复现：先在这里下结论，再决定要不要花钱跑端到端
    elif len(sys.argv) > 1 and sys.argv[1] == "final":
        final_main()              # Day14：只跑交付口径一组，落盘 eval_day14.json
    else:
        main()