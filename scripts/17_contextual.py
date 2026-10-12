"""实验 | W2 卡5：Contextual Retrieval（932 块语境前缀）+ token 成本账。

要回答的问题
    入库前给每块配一段"以它所在的整篇文档为背景"的简短语境说明，拼在正文前一起编码，
    召回层（recall / mrr / ndcg）涨多少、要花多少钱。

卡型：概念（前缀生成机制 + 缓存复用）+ 接线（建第二库 / 两格对照）

产出
    data/contextual_prefixes.jsonl     前缀缓存（可复用，重跑不重复烧钱）
    chroma_db 的 day6_kb_ctx           ctx 库（库名见 config.EMBED_MODELS[...]["variants"]["ctx"]）
    data/eval_w2_contextual.json       两格指标 + 延迟 + 成本 + 逐题明细

三段式用法（项目根 rag-kb-next 下，需 GPU + 关闭沙箱）
    # ① 小样：只跑 1 个 source 的前 32 块（约 0.1 元）—— 人工看前缀质量 + 读 usage 命中情况
    python scripts/17_contextual.py generate --source "RAG知识.pdf" --limit 32
    # ② 全量 932 块（约 1.8 元）—— 已生成的块按失效判据跳过，不重复烧钱
    python scripts/17_contextual.py generate
    # ③ 建 ctx 库 + 两格对照评测 + 成本账 + 两道闸
    python scripts/17_contextual.py build
    python scripts/17_contextual.py eval

实验纪律（沿用 scripts/16_model_ab.py，一项都不放松）
    1) 单一变量：两格只有 collection 不同（raw = 原文库 / ctx = 前缀库），代码路径完全一致。
       ⚠️ 不许出现 if cell 分支。
    2) 指标层：post-dedupe pre-quota（精排 + 去重后、配额截断前）。
       ⚠️ 两格精排必须读【同一份文本】（都读原文），否则变量变成两个（向量变了 + 精排输入变了）。
    3) 热态：warmup + 外层题 / 内层格交替，绝不"一格跑完再跑下一格"。
    4) 确定性：重跑基线格指标逐位相同；基线格 == W1 dense 组。

成本口径（config 三常量）
    cost = miss×PRICE_IN_PER_M + hit×PRICE_CACHE_HIT_PER_M + out×PRICE_OUT_PER_M（各 / 1e6）
    主项是【hit】（= 调用次数 × 整篇全文 token），不是 miss —— 命中价只有未命中的 1/50。
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包，运行时把项目根塞进 sys.path 才能 import rag.*（同 16_model_ab.py）
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rag.config import (  # noqa: E402
    BATCH_SIZE, GOLD_TUNE_JSON, LLM_MODEL, PRICE_CACHE_HIT_PER_M, PRICE_IN_PER_M,
    PRICE_OUT_PER_M, RECALL_K, RERANK_TOP_K, SOURCE_CAP, TESTSET_TUNE_JSONL,
    THINKING_DISABLED,
)
from rag.embedding import get_embedder  # noqa: E402
from rag.fusion import hybrid_retrieve  # noqa: E402
from rag.gold import load_gold, sha256_file  # noqa: E402
from rag.ingest import clean_meta, load_chunks  # noqa: E402
from rag.llm import get_llm, safe_preview  # noqa: E402
from rag.metrics import evaluate  # noqa: E402
from rag.retriever import BGEReranker, dedupe_adjacent, get_collection  # noqa: E402

PREFIX_CACHE_JSONL = ROOT / "data" / "contextual_prefixes.jsonl"
EVIDENCE_JSON = ROOT / "data" / "eval_w2_contextual.json"
W1_DENSE_JSON = ROOT / "data" / "eval_w1_ab.json"

# prompt 版本号：改了模板 / 换了 LLM / 改了粒度 → +1。它是缓存失效判据之一（见 TODO-2）
PROMPT_VERSION = "ctx-v1"

# 单变量：嵌入模型原地不动（卡4 已证 qwen3 无净增益），两格只差"块带不带语境前缀"
EMBED_KEY = "bge-small-zh"
GRID = {"raw": "raw", "ctx": "ctx"}      # 格名 → config 里的库变体名
BASELINE = "raw"
COLS = ["recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10"]
GROUP_LABEL = {
    "in": "in 题（页级 gold）",
    "cross": "cross 题（来源级 gold）",
    "partial": "partial 题（来源级 gold）",
    "all_with_gold": "全题混合（⚠️ 粒度/题型都不同，仅供参照）",
}


# ------------------------------------------------------------
# 已给：语料侧（source 全文 / 前缀缓存读写 / 入库）
# ------------------------------------------------------------
def sha256_text(s: str) -> str:
    """已给：字符串 sha256（前缀缓存判据用；文件级另有 gold.sha256_file）。"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def source_texts(chunks: list[dict]) -> dict[str, str]:
    """已给：{source: 整篇全文}。

    全文 = 该 source 的块按原顺序拼接（chunks.jsonl 的顺序就是切片顺序）。
    块间有 ~10% overlap，所以拼出来的文本比原始文档多约 10% 字 —— 只让 LLM prompt 变大
    （成本 +10%），不影响语义，故不去重（去重逻辑本身也会引入新错）。
    """
    grouped: dict[str, list[str]] = {}
    for c in chunks:
        grouped.setdefault(c["source"], []).append(c["text"])
    return {s: "\n".join(ts) for s, ts in grouped.items()}


def load_prefix_cache(path=PREFIX_CACHE_JSONL) -> dict[str, dict]:
    """已给：读前缀缓存，返回 {chunk_id: record}；文件不存在 → 空 dict（等于"全都没生成"）。"""
    if not Path(path).exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec
    return out


def save_prefix_cache(cache: dict[str, dict], path=PREFIX_CACHE_JSONL) -> None:
    """已给：原子写回（同目录临时文件 + os.replace）。

    为什么必须原子：生成 932 条要花真金白银，写到一半崩掉留下半截 jsonl 就得整体重跑。
    这是"真相源/花钱产物必须原子落盘"在缓存上的应用（同 ingest.save_chunks）。
    """
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in cache.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_ctx_index(collection, chunks: list[dict], cache: dict[str, dict]) -> int:
    """已给：把「语境前缀 + 正文」写进 ctx 库，返回写入后的总数。

    ⚠️ 不能复用 ingest.build_index：那一版 documents 写 c["text"]（原文），
    且 clean_meta 会把 "text" 键剥掉 —— 本卡要的恰好相反：
        documents           = 前缀 + "\\n" + 正文（编码输入，语境先于内容）
        metadata["raw_text"] = 原文（精排 / 评测 / preview 要读的那一份）
    键名绝不能叫 "text" —— 会被 clean_meta 静默剥掉，然后精排读到的是拼接串。
    """
    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in cache]
    if missing:
        raise AssertionError(
            f"{len(missing)} 块没有前缀（例如 {missing[:3]}）→ 先跑 generate"
        )
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        metas = []
        for c in batch:
            m = clean_meta(c)              # 剥掉 None 键与 "text"
            m["raw_text"] = c["text"]      # 原文另存；精排/评测读它
            metas.append(m)
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[cache[c["chunk_id"]]["prefix"] + "\n" + c["text"] for c in batch],
            metadatas=metas,
        )
    return collection.count()


# ------------------------------------------------------------
# 生成段（本卡核心：TODO-1 / TODO-2 / TODO-3）
# ------------------------------------------------------------
def context_prompt(source_text: str, chunk_text: str) -> str:
    """TODO-1：拼出"让 LLM 给这一块写语境说明"的 prompt。

    契约：
      输入  source_text —— 该块所在 source 的整篇全文（source_texts() 给的）
            chunk_text  —— 当前块的正文
      输出  str        —— 直接能当 chat.completions 的 user content

      硬约束（成本与质量全系于此）：
        · source_text 必须【整段】出现在 prompt 的【最前面】，逐字、逐 token 稳定 ——
          同一 source 的 337 次调用这一段必须一字不差，否则缓存命中恒为 0
          （未命中价 1.0 是命中价 0.02 的 50 倍，跑全量就是 ~1.8 元 vs ~90 元）。
        · 指令要明确要求输出 50–100 token 的【简短语境说明】，而不是复述 / 总结全文。

      自检（写完后自己回答再动手）：把 chunk_text 换成同一 source 的任意另一块，
      这个 prompt 的前 N 个字符是否一字未变？
    """
    return f"""你是一个文档语境分析助手。下面会给你同一篇 source 的全文，以及其中需要处理的一个块。请为这个块写一段简短的“语境说明”，用于帮助后续检索和正确理解该块。

        要求：
        1. 只依据给定的全文和当前块，不要编造未出现的信息。
        2. 说明当前块在全文中的位置、所属主题或章节，以及前后文正在讨论的核心内容。
        3. 明确当前块中关键实体、时间、条件，以及代词、省略或指代（如“它”“该方法”“上述方案”）在全文中的具体所指。
        4. 不要复述当前块全文，不要回答当前块中的问题，不要写无关总结。
        5. 用与全文相同的语言，输出 50-100 字。
        6. 只输出纯文本，不要加标题、前缀、解释或 Markdown 格式。
        
        【全文开始】
        {source_text}
        【全文结束】
        
        【当前块开始】
        {chunk_text}
        【当前块结束】

        语境说明："""


def prefix_is_valid(rec: dict, chunk: dict, source_sha: str) -> bool:
    """TODO-2：判断缓存里的这条前缀能否复用。

    契约：
      输入  rec        —— 缓存记录（load_prefix_cache 的一条）
            chunk      —— 当前 chunks.jsonl 里的这一块
            source_sha —— 该块所在 source 全文的 sha256（generate_prefixes 算好传入）
      输出  bool —— 三个判据全等才 True，任一不等 → 该条作废、重新生成

        ① chunk_id       块还在不在
        ② source_sha256  prompt 里带的是【整篇全文】，源一变，该 source 的【全部】前缀都该作废；
                         块级 hash 覆盖不了这一点（源变了但某块正文恰好没变）
        ③ prompt_version 改了模板 / 换了 LLM / 改了粒度 → +1；最不容易漏的一条

      ⚠️ 判据不足的后果是【静默】的：旧前缀被当新前缀用，A/B 结论失真却不报错
        （与 W1 卡3「BM25 token 缓存缺字段 → 静默失效」同族）。
    """
    # 判据全部用 .get()：旧记录缺字段时落到"不满足 → 重算"（安全方向），而不是 KeyError 崩掉。
    if rec.get("prompt_version") != PROMPT_VERSION:
        return False
    if rec.get("source_sha256") != source_sha:
        return False
    # 注：不需要再比 chunk_id —— "块还在不在"由调用方的 cache[chunk_id] 查表本身回答；
    # 在这里比 rec["chunk_id"] == chunk["chunk_id"] 是恒真式（rec 就是按这个 key 取出来的）。
    return True


def generate_prefixes(chunks: list[dict], cache: dict[str, dict],
                      source_texts_map: dict[str, str],
                      verbose: bool = False) -> tuple[dict, dict]:
    """TODO-3：生成 / 复用前缀，返回 (cache, stats)。

    契约：
      输入  chunks           —— 本次要处理的块（已由调用方按 --source / --limit 筛过）
            cache            —— 已有缓存 {chunk_id: rec}（就地更新，也允许返回新 dict）
            source_texts_map —— {source: 整篇全文}
            verbose          —— True 时把完整 prefix 打出来（小样人工检查用）
      输出  (cache, stats)
            stats = {"reused": int, "regenerated": int,
                     "miss_tokens": int, "hit_tokens": int, "out_tokens": int}
            ← 后三个【只统计本次真正调用的那些块】，供"本次花了多少"核对

      每条 rec 的字段：
            {"chunk_id", "source", "source_sha256", "prompt_version",
             "prefix", "miss_tokens", "hit_tokens", "out_tokens"}

      硬性要求：
        a) 【按 source 分组连续跑】，不按 chunks 原始顺序混着跑。
           理由不是"否则不命中"（缓存按前缀驻留，顺序不改变任一请求的前缀），
           而是把"前缀被淘汰 / 缓存过期"的风险压到最低，并让 usage 呈现成
           "每 source 1 次 miss + 其余全 hit"，账一眼能核对。
        b) 逐条从 usage 读三个【实测】字段记进 rec：
             prompt_cache_hit_tokens / prompt_cache_miss_tokens / completion_tokens
           ⚠️ 用实测值，不要用自己估的 token 数填表 —— 那等于自己给自己打分。
        c) 复用：prefix_is_valid 通过 → 直接沿用旧 rec（一次 LLM 都不调）；
           不通过 → 重新生成。
        d) 每块打一行进度（复用 / 重算、prefix 预览、本块 usage）。
        e) 结束前调 save_prefix_cache 落盘；中途崩掉也别丢已花的钱（可分批落盘）。

      调用形状（thinking 必须关，同 rag.llm.ask）：
            llm = get_llm()
            resp = llm.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": context_prompt(st, ct)}],
                temperature=0, extra_body=THINKING_DISABLED)
            resp.choices[0].message.content      → prefix
            resp.usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens / completion_tokens
    """
    llm = get_llm()
    # source 指纹在循环外算一次：全文是万级字符串，逐块重算 = 932 次无用哈希
    src_sha = {s: sha256_text(t) for s, t in source_texts_map.items()}
    stats = {"reused": 0, "regenerated": 0,
             "miss_tokens": 0, "hit_tokens": 0, "out_tokens": 0}

    # 按 source 分组连续跑（契约 a）：把"前缀被淘汰 / 缓存过期"的风险压到最低，
    # 并让 usage 呈现成"每 source 1 次 miss + 其余全 hit"，账一眼能核对。
    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk["source"], []).append(chunk)

    done = 0
    for src, group in by_source.items():
        s_text, s_sha = source_texts_map[src], src_sha[src]
        for chunk in group:
            cid = chunk["chunk_id"]
            rec = cache.get(cid)
            if rec and prefix_is_valid(rec, chunk, s_sha):
                stats["reused"] += 1
                done += 1
                print(f"[{done}/{len(chunks)}] 复用 {cid}  ({src})")
                if verbose:
                    print(f"    prefix: {rec['prefix']}")
                continue

            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": context_prompt(s_text, chunk["text"])}],
                temperature=0, extra_body=THINKING_DISABLED,
            )
            prefix = resp.choices[0].message.content
            u = resp.usage
            # usage 三字段用【实测值】（契约 b），不自己估 —— 估等于自己给自己打分。
            # 网关不返回缓存字段时（理论分支）按"全未命中"计：宁可高估，不可填 0 冒充真实数据
            #（Day11 的 chunks:0 教训同族）。
            hit = getattr(u, "prompt_cache_hit_tokens", None)
            miss = getattr(u, "prompt_cache_miss_tokens", None)
            if hit is None and miss is None:
                hit, miss = 0, u.prompt_tokens
            else:
                hit, miss = hit or 0, miss or 0
            out = u.completion_tokens or 0

            cache[cid] = {
                "chunk_id": cid, "source": src, "source_sha256": s_sha,
                "prompt_version": PROMPT_VERSION, "prefix": prefix,
                "miss_tokens": miss, "hit_tokens": hit, "out_tokens": out,
            }
            stats["regenerated"] += 1
            stats["miss_tokens"] += miss
            stats["hit_tokens"] += hit
            stats["out_tokens"] += out
            done += 1
            print(f"[{done}/{len(chunks)}] 重算 {cid}  hit={hit} miss={miss} out={out}  "
                  f"prefix={safe_preview(prefix, 60)}")
            if verbose:
                print(f"    prefix: {prefix}")
            if stats["regenerated"] % 20 == 0:     # 分批落盘（契约 e）：崩了也不丢已花的钱
                save_prefix_cache(cache)

    save_prefix_cache(cache)
    return cache, stats


# ------------------------------------------------------------
# 已给：成本账 / 512 截断体检
# ------------------------------------------------------------
def prefix_cost(miss: int, hit: int, out: int) -> dict:
    """已给：按 config 三常量算钱（元），分项返回 —— 让"主项是 hit"这件事看得见。"""
    c_miss = miss / 1e6 * PRICE_IN_PER_M
    c_hit = hit / 1e6 * PRICE_CACHE_HIT_PER_M
    c_out = out / 1e6 * PRICE_OUT_PER_M
    return {"miss_tokens": miss, "hit_tokens": hit, "out_tokens": out,
            "cost_miss": c_miss, "cost_hit": c_hit, "cost_out": c_out,
            "cost_total": c_miss + c_hit + c_out}


def print_cost(cache: dict[str, dict]) -> None:
    """已给：前缀生成成本账（全量总账 + hit 占比）。"""
    miss = sum(r.get("miss_tokens", 0) for r in cache.values())
    hit = sum(r.get("hit_tokens", 0) for r in cache.values())
    out = sum(r.get("out_tokens", 0) for r in cache.values())
    c = prefix_cost(miss, hit, out)
    total_in = miss + hit
    print(f"\n--- 前缀生成成本（{len(cache)} 条，口径：hit 价 = 未命中的 1/50）---")
    print(f"输入 miss={miss:>10,}  hit={hit:>12,}（命中率 {hit / total_in if total_in else 0:.1%}）"
          f"  out={out:>8,}")
    print(f"分项成本（元）：miss={c['cost_miss']:.4f}  hit={c['cost_hit']:.4f}  "
          f"out={c['cost_out']:.4f}  合计={c['cost_total']:.4f}")
    if c["cost_total"]:
        print(f"其中 hit 占 {c['cost_hit'] / c['cost_total']:.1%}，"
              f"miss 占 {c['cost_miss'] / c['cost_total']:.1%}（4 个 source 各一次冷启动）")


def print_token_check(chunks: list[dict], cache: dict[str, dict]) -> None:
    """已给：拼接串（前缀 + 正文）的实际 token 长度体检。

    ⚠️ bge 的 max_length=512 是【静默截断】：拼接串把语境说明放在最前，
    一旦超 512，被吃掉的是【正文的尾巴】—— 正文信息丢失且不报错。
    """
    tok = get_embedder(EMBED_KEY).tokenizer
    joined = [cache[c["chunk_id"]]["prefix"] + "\n" + c["text"] for c in chunks]
    lens = [len(x) for x in tok(joined, truncation=False, padding=False)["input_ids"]]
    over = [(chunks[i]["chunk_id"], n) for i, n in enumerate(lens) if n > 512]
    print(f"\n--- 512 截断体检（拼接串 token 长度，{len(lens)} 条）---")
    print(f"min={min(lens)} 中位={statistics.median(lens):.0f} max={max(lens)}  超 512 的 {len(over)} 条")
    for cid, n in over[:10]:
        print(f"  ⚠️ {cid}  {n} token → 正文尾部会被静默截掉")


# ------------------------------------------------------------
# 已给：检索段（两格走同一行代码 + 一处数据形状适配）
# ------------------------------------------------------------
def retrieve_stage(collection, ranker, question: str, variant: str):
    """已给：召回 → 精排 → 去重，返回 (hits, recall_ms, rerank_ms)。

    与 16_model_ab.py 同形，只有一处新增：把 hits 的 text 统一换成【原文】。
        raw 库  metadata 无 raw_text → 回退用 documents（本就等于原文）
        ctx 库  documents 是「前缀 + 正文」→ 用 metadata["raw_text"] 覆盖回来
    为什么必须统一（本卡单变量的要点）：指标层含精排，若 ctx 格精排吃拼接串、
    raw 格吃正文，两格差异就同时来自"向量变了"和"精排输入变了"，结论不成立。

    ⚠️ 这是数据形状适配（有无 raw_text 键），不是 if cell 分支 —— 两格执行同一行代码。
    variant 也必须逐格透传：它决定查询向量按哪一档截断（bge 两档同为 512，今天无差异，
    但漏传就是"改一处忘另一处"的种子）。
    """
    t0 = time.perf_counter()
    hits = hybrid_retrieve(collection, question, RECALL_K, mode="dense",
                           model_key=EMBED_KEY, variant=variant)
    for h in hits:
        if "raw_text" in h:
            h["text"] = h.pop("raw_text")
    t1 = time.perf_counter()
    ranked = ranker.rerank(question, hits)
    deduped = dedupe_adjacent(ranked)
    t2 = time.perf_counter()
    return deduped, (t1 - t0) * 1000, (t2 - t1) * 1000


def build_runners() -> dict:
    """已给：两格的 (collection, ranker)。两格共用同一个精排器实例。"""
    ranker = BGEReranker()
    return {g: (get_collection(model_key=EMBED_KEY, variant=GRID[g]), ranker) for g in GRID}


def build_rows(runners: dict, records: list[dict]):
    """已给：warmup + 外层题 / 内层格交替，返回 (rows_by_cell, lat_by_cell)。

    - warmup：records[0] 两格各跑一次，丢弃 / 不计时 / 不入表
    - 逐题交替：外层 for 题、内层 for 格（写成外层格会让后跑的格系统性受益）
    - gold 为空的题（out）：hits 照跑（延迟是真实流量），但不进 rows
    - 延迟口径 = 全部 20 题、两格同源
    """
    rows_by_cell: dict[str, list] = {g: [] for g in GRID}
    lat_by_cell: dict[str, list] = {g: [] for g in GRID}

    for g in GRID:
        retrieve_stage(runners[g][0], runners[g][1], records[0]["question"], GRID[g])

    for r in records:
        for g in GRID:
            hits, t_recall, t_rerank = retrieve_stage(
                runners[g][0], runners[g][1], r["question"], GRID[g])
            lat_by_cell[g].append((t_recall, t_rerank))
            if r["gold"]:
                rows_by_cell[g].append({**r, "hits": hits})   # {**r, ...} 防两格共享同一 dict
            n_hit = sum(1 for h in hits if h["chunk_id"] in r["gold"])
            print(f"{g:<4} id={r['id']:<3} 命中 {n_hit}/{len(r['gold'])}  "
                  f"召回 {t_recall:.1f}ms  精排+去重 {t_rerank:.1f}ms")

    return rows_by_cell, lat_by_cell


# ------------------------------------------------------------
# 已给：两道闸（确定性 + 基线回归）
# ------------------------------------------------------------
def determinism_check(runners: dict, records: list[dict], before: dict) -> None:
    """已给：重跑基线格，断言指标逐位相同（用 !=，不用 isclose）。"""
    key = BASELINE
    rows = [{**r, "hits": retrieve_stage(runners[key][0], runners[key][1],
                                        r["question"], GRID[key])[0]}
            for r in records if r["gold"]]
    after = evaluate(rows)

    diffs: list[str] = []
    for group, new_g in after["by_type"].items():
        old_g = before["by_type"].get(group)
        if old_g is None:
            diffs.append(f"by_type.{group}: 重跑后多出/少了这个题型")
            continue
        for col, v_new in new_g.items():
            v_old = old_g.get(col)
            if v_old != v_new:
                diffs.append(f"{group}.{col}: {v_old!r} → {v_new!r}")

    if diffs:
        print("确定性自检未通过：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"基线格重跑后指标漂移，共 {len(diffs)} 个字段")
    print("确定性自检通过：基线格重跑后指标【逐位相同】")


def baseline_regression_check(base_summary: dict) -> None:
    """已给：断言基线格 == W1 的 dense 组（卡5 没动 day6_kb，这是零偏差闸）。"""
    with open(W1_DENSE_JSON, "r", encoding="utf-8") as f:
        w1_dense = json.load(f)["metrics"]["dense"]

    diffs: list[str] = []
    for group, new_g in base_summary["by_type"].items():
        old_g = w1_dense["by_type"].get(group)
        if old_g is None:
            diffs.append(f"by_type.{group}: W1 里没有这个题型")
            continue
        for col, v_new in new_g.items():
            v_old = old_g.get(col)
            if v_old != v_new:
                diffs.append(f"{group}.{col}: W1={v_old!r} → 本次={v_new!r}")

    if diffs:
        print("基线回归未通过（raw 库被动过？）：")
        for d in diffs:
            print(f"  - {d}")
        raise AssertionError(f"基线格 != W1 dense 组，共 {len(diffs)} 个字段")
    print("基线回归通过：基线格 == W1 dense 组")


def latency_stats(pairs: list[tuple[float, float]]) -> dict:
    """已给：两段耗时的均值与中位数。"""
    rc = [p[0] for p in pairs]
    rr = [p[1] for p in pairs]
    return {"recall_mean_ms": statistics.fmean(rc), "recall_med_ms": statistics.median(rc),
            "rerank_mean_ms": statistics.fmean(rr), "rerank_med_ms": statistics.median(rr),
            "total_med_ms": statistics.median([a + b for a, b in pairs])}


def _pick(summary: dict, group: str) -> dict:
    return summary["all_with_gold"] if group == "all_with_gold" else summary["by_type"].get(group)


def print_table(metrics: dict, lat: dict) -> None:
    """已给：指标对照表（每题型一组）+ 延迟表。"""
    print(f"\n参数：RECALL_K={RECALL_K} RERANK_TOP_K={RERANK_TOP_K}"
          f"（本表【未截断】）SOURCE_CAP={SOURCE_CAP} mode=dense embed={EMBED_KEY}")
    print(f"指标层级：精排 + 去重后全量 | 深度 @3/@5/@10 | 执行：warmup + 逐题交替")
    print(f"对照轴：块带不带语境前缀 | 基线格 = {BASELINE}（原文库）")

    for group in ("in", "cross", "partial", "all_with_gold"):
        base = _pick(metrics[BASELINE], group)
        if not base:
            continue
        print(f"\n--- {GROUP_LABEL[group]} ---")
        print(f"{'格':<8}{'n':>3} " + "".join(f"{c:>10}" for c in COLS))
        for cell in GRID:
            g = _pick(metrics[cell], group)
            print(f"{cell:<8}{g['n_scored']:>3} "
                  + "".join(f"{g.get(c, float('nan')):>10.4f}" for c in COLS))
        print(f"{'  (Δ)':<8}{'':>3} " + "".join(
            f"{_pick(metrics[c], group).get(col, 0) - base.get(col, 0):>+10.4f}"
            if c != BASELINE else " " * 10
            for c in GRID for col in COLS))

    print("\n--- 延迟（全部 20 题，两格同源）---")
    print(f"{'格':<8}{'召回均值':>10}{'召回中位':>10}{'精排均值':>10}{'精排中位':>10}{'合计中位':>10}")
    for cell in GRID:
        l = lat[cell]
        print(f"{cell:<8}{l['recall_mean_ms']:>10.1f}{l['recall_med_ms']:>10.1f}"
              f"{l['rerank_mean_ms']:>10.1f}{l['rerank_med_ms']:>10.1f}{l['total_med_ms']:>10.1f}")


def dump_evidence(metrics: dict, lat: dict, rows_by_cell: dict, cache: dict) -> None:
    """已给：落盘（指纹可溯源 + 逐题明细，免重跑 GPU / 免重烧钱）。"""
    per_question = {}
    for cell, rows in rows_by_cell.items():
        per_question[cell] = [
            {"id": r["id"], "type": r["type"], "unit": r["unit"], "n_gold": len(r["gold"]),
             "hit_ranks": [i for i, h in enumerate(r["hits"], 1) if h["chunk_id"] in r["gold"]],
             "hits": [{"rank": i, "chunk_id": h["chunk_id"],
                       "rerank_score": round(h.get("rerank_score", 0.0), 4),
                       "in_gold": h["chunk_id"] in r["gold"], "preview": h["text"][:80]}
                      for i, h in enumerate(r["hits"], 1)]}
            for r in rows
        ]

    miss = sum(r.get("miss_tokens", 0) for r in cache.values())
    hit = sum(r.get("hit_tokens", 0) for r in cache.values())
    out = sum(r.get("out_tokens", 0) for r in cache.values())
    payload = {
        "experiment": "w2_contextual_retrieval",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "RECALL_K": RECALL_K, "RERANK_TOP_K": RERANK_TOP_K, "SOURCE_CAP": SOURCE_CAP,
            "mode": "dense", "embed_key": EMBED_KEY, "metric_layer": "post_dedupe_pre_quota",
            "grid": {g: {"variant": GRID[g]} for g in GRID}, "baseline": BASELINE,
            "prompt_version": PROMPT_VERSION, "prefix_granularity": "source",
        },
        "fingerprints": {
            "testset": TESTSET_TUNE_JSONL.name, "testset_sha256": sha256_file(TESTSET_TUNE_JSONL),
            "gold": GOLD_TUNE_JSON.name, "gold_sha256": sha256_file(GOLD_TUNE_JSON),
            "prefix_cache": PREFIX_CACHE_JSONL.name,
            "prefix_cache_sha256": sha256_file(PREFIX_CACHE_JSONL) if PREFIX_CACHE_JSONL.exists() else None,
        },
        "metrics": metrics,
        "latency": lat,
        "cost": prefix_cost(miss, hit, out),
        "cost_prices": {"in_per_m": PRICE_IN_PER_M, "cache_hit_per_m": PRICE_CACHE_HIT_PER_M,
                        "out_per_m": PRICE_OUT_PER_M},
        "per_question": per_question,
    }
    with open(EVIDENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘：{EVIDENCE_JSON}")


# ------------------------------------------------------------
# 子命令
# ------------------------------------------------------------
def cmd_generate(args) -> None:
    """已给：筛块 → 生成 / 复用前缀 → 落盘 → 打印本次花费。"""
    all_chunks = load_chunks()
    st = source_texts(all_chunks)          # ⚠️ 用【全量】算全文：--limit 只筛要处理的块
    todo = all_chunks
    if args.source:
        todo = [c for c in todo if c["source"] == args.source]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        raise SystemExit(f"--source {args.source!r} 没匹配到任何块")

    cache = load_prefix_cache()
    print(f"待处理 {len(todo)} 块（全库 {len(all_chunks)} 块，{len(st)} 个 source）；"
          f"已有缓存 {len(cache)} 条；prompt_version={PROMPT_VERSION}")
    cache, stats = generate_prefixes(todo, cache, st, verbose=bool(args.limit))
    save_prefix_cache(cache)

    c = prefix_cost(stats["miss_tokens"], stats["hit_tokens"], stats["out_tokens"])
    print(f"\n本次：复用 {stats['reused']} / 重算 {stats['regenerated']}")
    print(f"本次 token：miss={c['miss_tokens']:,} hit={c['hit_tokens']:,} out={c['out_tokens']:,}")
    print(f"本次花费：{c['cost_total']:.4f} 元"
          f"（miss {c['cost_miss']:.4f} / hit {c['cost_hit']:.4f} / out {c['cost_out']:.4f}）")
    print(f"缓存已落盘：{PREFIX_CACHE_JSONL}（{len(cache)} 条）")


def cmd_build(args) -> None:
    """已给：建 ctx 库（documents = 前缀 + 正文，metadata 存 raw_text）。"""
    chunks = load_chunks()
    cache = load_prefix_cache()
    col = get_collection(model_key=EMBED_KEY, variant=GRID["ctx"])
    before = col.count()
    n = build_ctx_index(col, chunks, cache)
    print(f"ctx 库（{GRID['ctx']} → {col.name}）写入前 {before} 条，写入后 {n} 条；"
          f"chunks.jsonl {len(chunks)} 条")
    if n != len(chunks):
        raise AssertionError(f"ctx 库条数 {n} != chunks {len(chunks)}")


def cmd_eval(args) -> None:
    """已给：两格对照 → 指标/延迟/成本 → 两道闸 → 落盘。"""
    chunks = load_chunks()
    cache = load_prefix_cache()
    ctx_col = get_collection(model_key=EMBED_KEY, variant=GRID["ctx"])
    if ctx_col.count() != len(chunks):
        raise AssertionError(f"ctx 库只有 {ctx_col.count()} 条、chunks 有 {len(chunks)} 条"
                             f"→ 先跑全量 generate 再 build（小样库不能用来评测）")

    records = load_gold()["records"]
    runners = build_runners()
    rows_by_cell, lat_by_cell = build_rows(runners, records)
    metrics = {g: evaluate(rows_by_cell[g]) for g in GRID}
    lat = {g: latency_stats(lat_by_cell[g]) for g in GRID}

    print_table(metrics, lat)
    print_cost(cache)
    print_token_check(chunks, cache)

    determinism_check(runners, records, metrics[BASELINE])
    baseline_regression_check(metrics[BASELINE])
    dump_evidence(metrics, lat, rows_by_cell, cache)


def main() -> None:
    ap = argparse.ArgumentParser(description="W2 卡5：Contextual Retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="生成 / 复用 932 条语境前缀")
    p.add_argument("--source", default=None, help="只跑这个 source（小样用）")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 块（小样用）")
    p.set_defaults(func=cmd_generate)

    sub.add_parser("build", help="建 ctx 库").set_defaults(func=cmd_build)
    sub.add_parser("eval", help="两格对照评测 + 成本账 + 两道闸").set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()