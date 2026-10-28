# W5 轨迹统计（一页）

数据源：`data/w4_agentic_traces_full.jsonl`（100 条 = 20 题 × 5 组），脚本 `scripts/24_trajectory_stats.py` 只读、零 API、可重跑复现。

## 1. 检索轮次分布（Agentic 组）

| 组 | 平均轮次 | 0 轮 | 1 轮 | 2 轮 | 3 轮（触顶） | 触顶题 |
|---|---|---|---|---|---|---|
| C | 1.40 | 2 | 11 | 4 | 3 | id=10(out)、id=22(partial)、id=27(out) |
| D1 | 1.50 | 2 | 10 | 4 | 4 | id=9(out)、id=10(out)、id=22(partial)、id=27(out) |

参照：A 组固定 1 轮、B 组固定 2 轮（无自评，不参与提前终止）。

按题型（C 组）：

| 题型 | n | 平均轮次 | 触顶 |
|---|---|---|---|
| in | 8 | 1.00 | 0 |
| cross | 5 | 0.80 | 0 |
| partial | 3 | 2.33 | 1 |
| out | 4 | 2.25 | 2 |

## 2. 终止原因分布

| 组 | self_eval 足够 | 触顶 | gate 不检索 | 改写重复 | 缓存命中 |
|---|---|---|---|---|---|
| A | 0 | 20 | 0 | 0 | 0 |
| B | 0 | 20 | 0 | 0 | 0 |
| C | 13 | 3 | 2 | 2 | 0 |
| D1 | 13 | 4 | 2 | 1 | 0 |
| D2 | 0 | 0 | 2 | 0 | 18 |

注：A/B 的「触顶」= 固定管线跑完预设轮次（不是撞上 step cap）；C/D1 的 0 轮 = gate 判「不检索」。C 组 cross 平均轮次被 2 道 0 轮题拉低。

## 3. 改写有效率（分母 = 改写轮，不是题数）

| 组 | 改写轮 | 与上轮重复 | 引入新块 | 引入新 gold |
|---|---|---|---|---|
| B | 20 | 0/20 = 0.0% | 11/20 = 55.0% | 1/20 = 5.0% |
| C | 12 | 2/12 = 16.7% | 5/12 = 41.7% | 0/12 = 0.0% |
| D1 | 13 | 1/13 = 7.7% | 8/13 = 61.5% | 1/13 = 7.7% |

**合计**：改写轮 45，其中与上轮重复 3；引入新 top-3 块 24/45 = 53.3%；引入新 gold 2/45 = 4.4%。

## 4. 命中去向（D2 缓存）

- D1 写入缓存 18/20 题；D2 命中 18/20 题（写过的题 100% 命中）
- 命中趟延迟：均值 7.50ms / 中位 7.22ms / 范围 4.89–11.16ms，llm_calls=0
- 相似度范围 0.999999–1.000001（原题重复，无假命中）
- 结构性未命中 2 题：id=7(gate_no_retrieve)、id=11(gate_no_retrieve)

## 附录：每题轨迹（C / D1）

| 组 | id | 题型 | 轮次 | 终止 | 每轮 top1 |
|---|---|---|---|---|---|
| C | 1 | in | 1 | self_eval_sufficient | RAG知识_p015_c004 |
| C | 2 | in | 1 | self_eval_sufficient | RAG知识_p013_c002 |
| C | 3 | in | 1 | self_eval_sufficient | RAG知识_p015_c002 |
| C | 4 | in | 1 | self_eval_sufficient | RAG知识_p021_c003 |
| C | 5 | in | 1 | self_eval_sufficient | RAG知识_p021_c001 |
| C | 7 | cross | 0 | gate_no_retrieve |  |
| C | 8 | cross | 1 | self_eval_sufficient | RAG 技术概览_web_c025 |
| C | 9 | out | 2 | query_repeat | Transformer--Attention Is All You Need_web_c008 → Transformer--Attention Is All You Need_web_c013 |
| C | 10 | out | 3 | cap | Transformer--Attention Is All You Need_web_c010 → Transformer--Attention Is All You Need_web_c011 → Transformer--Attention Is All You Need_web_c007 |
| C | 11 | cross | 0 | gate_no_retrieve |  |
| C | 12 | in | 1 | self_eval_sufficient | RAG知识_p024_c001 |
| C | 15 | in | 1 | self_eval_sufficient | RAG知识_p030_c001 |
| C | 16 | in | 1 | self_eval_sufficient | RAG知识_p123_c003 |
| C | 18 | cross | 2 | self_eval_sufficient | RAG知识_p030_c004 → RAG知识_p070_c002 |
| C | 21 | cross | 1 | self_eval_sufficient | RAG知识_p087_c001 |
| C | 22 | partial | 3 | cap | RAG 技术概览_web_c037 → RAG 技术概览_web_c038 → RAG 技术概览_web_c038 |
| C | 23 | partial | 2 | query_repeat | RAG知识_p016_c001 → RAG知识_p131_c001 |
| C | 25 | partial | 2 | self_eval_sufficient | RAG 技术概览_web_c030 → RAG 技术概览_web_c030 |
| C | 27 | out | 3 | cap | Transformer--Attention Is All You Need_web_c093 → Transformer--Attention Is All You Need_web_c093 → Transformer--Attention Is All You Need_web_c093 |
| C | 30 | out | 1 | self_eval_sufficient | Redis全套学习笔记_p110_c003 |
| D1 | 1 | in | 1 | self_eval_sufficient | RAG知识_p015_c004 |
| D1 | 2 | in | 1 | self_eval_sufficient | RAG知识_p013_c002 |
| D1 | 3 | in | 1 | self_eval_sufficient | RAG知识_p015_c002 |
| D1 | 4 | in | 1 | self_eval_sufficient | RAG知识_p021_c003 |
| D1 | 5 | in | 1 | self_eval_sufficient | RAG知识_p021_c001 |
| D1 | 7 | cross | 0 | gate_no_retrieve |  |
| D1 | 8 | cross | 1 | self_eval_sufficient | RAG 技术概览_web_c025 |
| D1 | 9 | out | 3 | cap | Transformer--Attention Is All You Need_web_c008 → Transformer--Attention Is All You Need_web_c008 → Transformer--Attention Is All You Need_web_c008 |
| D1 | 10 | out | 3 | cap | Transformer--Attention Is All You Need_web_c010 → Transformer--Attention Is All You Need_web_c011 → Transformer--Attention Is All You Need_web_c011 |
| D1 | 11 | cross | 0 | gate_no_retrieve |  |
| D1 | 12 | in | 1 | self_eval_sufficient | RAG知识_p024_c001 |
| D1 | 15 | in | 1 | self_eval_sufficient | RAG知识_p030_c001 |
| D1 | 16 | in | 1 | self_eval_sufficient | RAG知识_p123_c003 |
| D1 | 18 | cross | 2 | self_eval_sufficient | RAG知识_p030_c004 → RAG知识_p070_c002 |
| D1 | 21 | cross | 1 | self_eval_sufficient | RAG知识_p087_c001 |
| D1 | 22 | partial | 3 | cap | RAG 技术概览_web_c037 → RAG 技术概览_web_c038 → RAG 技术概览_web_c038 |
| D1 | 23 | partial | 2 | query_repeat | RAG知识_p016_c001 → RAG知识_p131_c001 |
| D1 | 25 | partial | 2 | self_eval_sufficient | RAG 技术概览_web_c030 → RAG 技术概览_web_c030 |
| D1 | 27 | out | 3 | cap | Transformer--Attention Is All You Need_web_c093 → Transformer--Attention Is All You Need_web_c093 → Transformer--Attention Is All You Need_web_c003 |
| D1 | 30 | out | 2 | self_eval_sufficient | Redis全套学习笔记_p110_c003 → Redis全套学习笔记_p110_c003 |

口径：轨迹只落 top-3；gold 按题粒度（in=页级块 / cross·partial=来源级）；「新块」= 新进 top-3 的块，不是新被召回的块。
