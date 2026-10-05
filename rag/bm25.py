"""检索层 | W1 卡3：BM25 稀疏检索（jieba 分词 + rank_bm25）。

为什么要它：dense（bi-encoder）把 query 和 doc 各自压成向量，语义泛化强，
但对【字面 token】不敏感 —— 语料里的 Vectorize / chunkText / RRF 这类拉丁串
在中文语义空间里被稀释。BM25 是词袋模型，只问「query 里的词在这块里出现过吗」。
两路失败模式不相关，这正是 RRF 融合能涨的前提（讲义第 3 节）。

本文件三条硬约束（讲义第 5 节）：
  1) BM25Okapi 的输入是【已分词的 token list】，不是原始字符串；中文必须 jieba
  2) jieba 实测会把整段拉丁串保留为单个 token（chunkText / RRF / Vectorize 都没被切碎），
     所以真正的风险不是「切碎」，而是 BM25 只做 token 精确相等 ——
     query 的 "chunk" 命中不了文档的 "chunkText"（前缀关系），加词典也修不好
  3) 分词结果必须落盘缓存，且缓存必须能【判失效】。不缓存会让每次评测重切 932 块
     （jieba 首次加载词典 ~1s），检索延迟一列直接作废

用法（项目根 rag-kb-next 下，包内代码一律 python -m）：
    python -m rag.bm25        # 自检：建索引 + 打印 3 个样例 query 的 top-5
"""
import json
import logging

import jieba
from rank_bm25 import BM25Okapi

from .config import BM25_B, BM25_K1, BM25_TOKENS_CACHE, CHUNKS_JSONL


# ------------------------------------------------------------
# 技术词词典（TODO-2）
# ------------------------------------------------------------
# ⚠️ 骨架里原来的提示「扫一遍语料里的拉丁串」是【错的，已作废】：实测扫出 3573 个候选，
#    根本筛不过来；而且方向本身就不对。2026-10-02 实测证据（自己复核一遍）：
#
# A. 纯拉丁串：add_word 是【空操作】，一个都不用加。加词前后完全一样：
#      '使用RRF做融合检索'            -> ['使用','RRF','做','融合','检索']      （前后同）
#      'chunkText先按段落切'          -> ['chunkText','先','按','段落','切']    （前后同）
#      '把Vectorize索引的维度设为768' -> ['把','Vectorize','索引','的','维度','设为','768']（前后同）
#    → jieba 本来就把整段拉丁保留成单个 token，加词典改变不了任何东西。
#    → 顺带否掉一个念头：「刻意让 chunkText 裂成 chunk + Text」在当前实现下【做不到】。
#      所以「打 chunk 命中 chunkText」这条前缀路径救不回来 —— 放弃它，交给 dense。
#
# B. add_word 真正生效的只有两处：
#    B1 中文领域术语（主战场）：
#         '向量数据库' -> ['向量','数据库']      加词后 -> ['向量数据库']
#         '交叉编码器' -> ['交叉','编码器']      加词后 -> ['交叉编码器']
#         '上下文检索' -> ['上下文','检索']      加词后 -> ['上下文检索']
#       ← 默认切出的两个泛词会把 IDF 摊平，「向量数据库」作为整体概念的区分度直接没了。
#    B2 带连字符 / 下划线的标识符（数量少，手工列得完）：
#         'bge-small-zh'      -> ['bge','-','small','-','zh']      加词后整词
#         'top_k'             -> ['top','_','k']                   加词后整词
#         'bge-reranker-base' -> ['bge','-','reranker','-','base'] 加词后整词
#
# C. 筛选标准：按【query 驱动】，不按【语料驱动】。
#    BM25 的收益只体现在 query 里真的出现的词上；语料里只出现一两次的长尾术语
#    加进去对召回零贡献，只会稀释维护成本。所以别去把 3573 个拉丁串全过一遍。
#    落地办法：先留空，跑一遍 tune 集 20 题看哪些 query 的领域术语被切坏，再补。
# 【你的原始判断（10-02，已证伪，留档不删）】
#   「用户打字倾向简写，较难完整打出 chunkText 这类词，故需要 chunk 也能与 chunkText 匹配」
#   —— 问题意识是对的（用户确实会简写），但解法不成立：见上面 A 段，
#      add_word 改不了拉丁串的切分，「让 chunkText 裂开」在当前实现下做不到。
#   留档理由：「意识到问题」≠「找到了可行的解法」，这两件事要能分开评价。
TECH_WORDS: list[str] = [
    # TODO-2：按 B1（中文领域术语为主）+ B2（连字符/下划线标识符）补词。
    # 判据：「这个词被切碎后，会不会让一个整体概念退化成两个泛词？」
    # 干扰项：纯英文单词（Answer / Array / All）、人名（Aaron / Andrew）——
    #        它们不影响 BM25 的召回，不要加。
]


# ------------------------------------------------------------
# 停用词（只放【已被实测证明】会污染打分的词）
# ------------------------------------------------------------
# 依据：计划文档「W1 实测修正」第 6 条 —— rank_bm25 的 IDF 用 ATIRE 变体
#   idf = log((N - n + 0.5) / (n + 0.5))，【没有】经典 BM25 那个 +1。
# 后果：高频词的 idf 会算成【负数】，而负 idf 又被 epsilon 下限兜成【正数 +1.371】
#   （epsilon = 0.25 × average_idf = 0.25 × 5.4839）。
# 实测全库只有 3 个 token 被兜住：'\n' / '的' / '：'（其余 6518 个都是正的）。
# ⇒ 「IDF 自动压掉高频词」这句话在 rank_bm25 上【不成立】：
#   '\n' 被不变式滤掉了，'：' 也是标点；只有 '的' 漏网 ——
#   它给全库几乎每一块都加了一个正偏移，命中与否不影响，纯属噪声投票。
# 注意这是【数据决定的特例】，不是「照抄英文停用词表」：
# 默认口径仍是「IDF 自己会压高频词，别手删」；每加一个词都要有实测依据。
STOPWORDS: set[str] = {"的"}


# jieba 初始化：【模块级执行一次】。
# 原来这段放在 tokenize() 里 —— 每切一段文本就重加一遍词典，幂等、不出错，
# 但把「全局副作用」藏进了本该是纯函数的 tokenize 里，读代码的人会误判它可以随便调。
for _w in TECH_WORDS:
    jieba.add_word(_w)

# 静音 jieba 的 "Building prefix dict..." / "Loading model cost..." 日志
jieba.setLogLevel(logging.ERROR)


def _is_noise(token: str) -> bool:
    """不变式：token 里一个字母 / 数字 / 汉字都没有 → 纯标点或空白，是噪声。

    为什么用不变式不用黑名单：黑名单永远会漏 —— 原来那份 waste_list 就漏掉了
    '\\n'（全库第一高频 token，8002 次，占全部 103615 个 token 的 7.7%）。
    不变式一次全覆盖：' ' / '\\n' / '( ) , 。 ？' 全部判为噪声，将来语料里
    冒出新的标点也不会漏。
    注：str.isalnum() 对汉字也返回 True（CJK 属 Unicode 字母类），中英一网打尽。
    """
    return not any(ch.isalnum() for ch in token)


def tokenize(text: str) -> list[str]:
    """TODO-3：把一段原文切成 token list。query 和 doc 都走这一个函数。

    为什么必须是同一个函数：两路用不同分词器，融合出问题时你会分不清
    是「融合的锅」还是「分词的锅」。这也是卡4 能定位问题的前提。

    要求（每条都要能说出为什么）：
      - 先 text.lower() 再切 —— BM25 对大小写敏感，query 写 "rrf" 而库里是 "RRF"
        会直接漏召，而这不是模型的错，是你没归一化
      - 用 jieba.lcut(text) —— 精确模式。别用 cut_for_search：
        搜索引擎模式会把长词再切出子串，等于人为放大词频（讲义第 2 节① 刚讲过
        BM25 已经在压词频了，你再手动放大一次是自相矛盾）
      - 过滤什么：走 _is_noise 不变式（见上）。黑名单版本漏了 '\n'，
        而 '\n' 是全库第一高频 token —— 等于让最大的一坨噪声一路进到打分里。
      - 停用词：**只删 STOPWORDS 里那个 '的'**，理由写在常量定义处（有实测依据）。
        别照抄英文停用词表 —— 默认口径仍是「IDF 自己会压高频词」，
        手删的每一个词都是「用确定的损失换不确定的收益」。
    """
    text = text.lower()
    tokens = jieba.lcut(text)
    return [t for t in tokens if not _is_noise(t) and t not in STOPWORDS]


def load_chunks() -> list[dict]:
    """读真相源 chunks.jsonl"""
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tokens(chunks: list[dict]) -> list[list[str]]:
    """TODO-4：读分词缓存；缓存缺失或失效就重切并覆盖写回。

    缓存落盘 BM25_TOKENS_CACHE（jsonl，一行一块：{"chunk_id": ..., "tokens": [...]}）

    失效判据必须是【显式判断】，不能只 return 文件存在就完事：
      - 缓存的 chunk_id 序列与当前 chunks 逐一比对（顺序 + 内容）
      - 块数不等
    判据不通过 → 重切 → 覆盖写回，并打印一行「缓存失效，已重建 N 块」。

    返回：与 chunks 等长、顺序一一对应的 token list
    （rank_bm25 靠【下标】对齐语料，顺序错一格，召回结果就整体错位，且不会报错）
    """
    tokens_list = []
    try:
        with open(BM25_TOKENS_CACHE, "r", encoding="utf-8") as f:
            tokens_dict = [json.loads(line) for line in f if line.strip()]
        # 缓存正确
        if len(tokens_dict) == len(chunks):
            if all(d1["chunk_id"] == d2["chunk_id"] for (d1, d2) in zip(chunks, tokens_dict)):
                return [t["tokens"] for t in tokens_dict]
        # 重建缓存
        tokens_list = [tokenize(c["text"]) for c in chunks]
        print(f"缓存失效，已重建 {len(tokens_list)} 块")
    except FileNotFoundError:
        tokens_list = [tokenize(c["text"]) for c in chunks]
        print(f"新建缓存文件，路径：{BM25_TOKENS_CACHE}")
    # 先建目录：首次运行（或缓存被手工删掉）时 processed/ 可能不存在，
    # 不 mkdir 会直接 FileNotFoundError，而且报错点落在 write 而不是 read，容易看懵
    BM25_TOKENS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_TOKENS_CACHE, "w", encoding="utf-8") as f:
        for c, toks in zip(chunks, tokens_list):
            f.write(json.dumps(
                {"chunk_id": c["chunk_id"], "tokens": toks},
                ensure_ascii=False,
            ) + "\n")
    return tokens_list

_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def get_index() -> tuple[BM25Okapi, list[dict]]:
    """已给：懒加载 BM25 索引（进程内只建一次），返回 (bm25, chunks)。

    为什么懒加载 + 进程内缓存：卡7 的对照表要跑 20 题 × 2 组，每题都重建索引，
    光分词就能把延迟指标淹掉。模块级变量保证「同一进程里只建一次」。

    BM25Okapi(corpus, k1=..., b=...) 里的 corpus 是【token list 的列表】。
    """
    global _bm25, _chunks
    if _bm25 is None:
        _chunks = load_chunks()
        _bm25 = BM25Okapi(load_tokens(_chunks), k1=BM25_K1, b=BM25_B)
    return _bm25, _chunks


def bm25_search(query: str, k: int = 10) -> list[dict]:
    """TODO-5：BM25 召回，返回与 retriever.retrieve() 【同构】的 hits。

    ○ 机制（已实测，不用猜）：
      - bm25.get_scores(query_tokens) 返回 numpy.ndarray，长度 = 全库 932，【不是 top-k】。
        ⚠️ 骨架原注释写「它内部除以了 len(query)，按词数取平均」——【错，已更正】。
        源码是纯求和（`return score`，没有除法）。后果很重要：query token 越多、
        命中越多，分数就越大 —— 所以【分数绝对值不能跨 query 比较】。
        实测「怎么把长文本切成小段」（8 个 token）拿 17.350，
        比「RRF 融合」（2 个 token）的 11.364 还高，但它的 top-5 全是无关块。
        卡7 里若出现「BM25 分高所以它准」的推论，直接否掉。
      - 取 top-k 的惯用法：order = scores.argsort()[::-1][:k]
        argsort() 给你「升序排列对应的下标」，[::-1] 翻成降序，[:k] 取前 k 个。
        为什么现在【不该】上 np.argpartition：它 O(n) 但只保证「前 k 个是最大的」、
        不保证有序，你还得再排一次；932 这个规模 argsort 是微秒级，
        提前优化只会让代码更难读。
      - 【空 query 保护】query 若被 tokenize 整个滤光（例如用户只打了个标点），
        get_scores([]) 会除零得到 nan。要早返回 []。

    ○ 0 分块的决策（实测数据决定了你怎么写）：
      rank_bm25 的 IDF 是 log((N - n + 0.5)/(n + 0.5))，【没有】经典公式里那个 +1。
      后果是 idf 可以为 0。实测（N=10）：
          idf['rag'] = 1.2238 (n=2)      idf['mid'] = 0.0 (n=5 = N/2)
          query=['rag'] -> score>0 的块 [0,1]      score==0 的块：其余 8 块
          query=['mid'] -> score>0 的块 []         score==0 的块：全部 10 块
      ⇒「0 分」有两种来源：① 一个词都没命中；② 只命中了 idf=0 的中频词。
        第二种情况里，一个【真的命中了 query 词】的块也会是 0 分 —— 这个坑要知道，
        否则将来遇到「明明命中却排不上」会查半天。
      ⇒ 决策方向仍是过滤掉 score <= 0（idf=0 的词不带区分度，放它进 RRF 等于给噪声投票），
        但理由要自己讲清楚。留个问题给你：不过滤会怎样？
        提示：argsort 对一堆 0 分的先后是【任意顺序】（quicksort 不稳定），于是「top-5」
        会变成 932 块里随便挑 5 块，每块稳拿 1/(60+rank) ≈ 0.016 的融合分 ——
        而真正的第 1 名也只有 1/61 ≈ 0.0164。这等于给噪声发满额选票。

    步骤：
      1) query 走同一个 tokenize，并处理空 token 的情况
      2) scores = bm25.get_scores(tokens)
      3) 取 top-k 的【下标】（见上）
      4) 过滤掉 score <= 0
      5) 组装 hits —— 【字段别省】：chunk_id / score / text / source / page。
         下游 format_sources 读 source/page，expect_hit（页级 gold）也读 source/page。
         少字段不报错，只会让指标静默变 0。
         ⚠️ 实测坑：chunks.jsonl 的字段是【扁平】的
             ['chunk_id','doc_id','source','page','url','title','text','char_count']
         并没有 meta 子字典 —— 所以 **chunk.get("meta", {}) 永远展开成空，
         doc_id / url / title / char_count 会静默丢掉（不报错，最难查）。
         正确写法：hit = dict(chunk) 整块平铺，再补 hit["score"]。

    自检口径（跑 python -m rag.bm25 时应能看到）：
      - "RRF 融合" 这种【原样出现】的拉丁 token → 命中且分数明显更高
      - "chunkText 的分块策略" → 能命中含 chunkText 的块（整词相等）
      - "怎么把长文本切成小段" 这种纯语义改写 → 命中很差，top-5 里多是靠「文本/切/段」
        这类泛词刷进来的，分数低且分散 —— 这正是讲义第 3 节说的「BM25 帮不上忙」那一格
    """
    # TODO-5(a)：空 query 保护。tokenize 可能把 query 整个滤光
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # 索引和语料都要拿到（_chunks 的下标就是 get_scores 的下标，两者天然对齐）
    bm25, chunks = get_index()
    scores = bm25.get_scores(query_tokens)      # 相关性分数

    # TODO-5(b)：取 top-k 的【下标】，按分数降序
    order = scores.argsort()[::-1][:k]          # argsort()返回排序后原数组的下标顺序

    # TODO-5(c)：遍历 order，过滤掉 score <= 0，组装 hits
    hits = []
    for i in order:
        score = scores[i]
        if score <= 0:
            continue

        chunk = chunks[i]
        # 整块平铺（chunks.jsonl 是扁平 schema，没有 meta 子字典可展开），
        # 再补上 BM25 分。float(score) 是必要的：numpy.float32 直接进 json.dumps
        # 会 TypeError，trace.py 落盘那一层会炸。
        hit = dict(chunk)
        hit["score"] = float(score)
        hits.append(hit)

    return hits


if __name__ == "__main__":
    # 已给：自检入口。三件事要看得到：
    #   ① 拉丁 token 原样 query 命中 ② 中文 query 能召回 ③ 纯语义改写 query 明显失效
    bm25, chunks = get_index()
    print(f"BM25 索引就绪：{len(chunks)} 块，k1={BM25_K1}, b={BM25_B}")
    for q in ("RRF 融合", "chunkText 的分块策略", "怎么把长文本切成小段"):
        print(f"\nquery: {q}")
        for h in bm25_search(q, 5):
            # 块文本里含换行（PDF 抽出来的原文），不压平会把一次自检打成几十行
            snippet = h["text"][:36].replace("\n", " ")
            print(f"  {h['score']:>8.3f}  {h['chunk_id']:<22} {snippet}")