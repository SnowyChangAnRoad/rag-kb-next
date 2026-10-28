# W3 收尾报告 · MCP server 化 + 工具描述对比 + 安全加固

生成时间：2026-10-14
对应计划：`docs/RAG深度强化阶段计划.md` W3（[L658–679](file:///c:/Users/xx/Desktop/python/Agent%E5%AD%A6%E4%B9%A0/RAG/rag-kb-next/docs/RAG%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E9%98%B6%E6%AE%B5%E8%AE%A1%E5%88%92.md)）
全局计划口径：`Agent开发学习计划.md`（W3 = MCP Server 化 + Injection；「MCP 深入」显式排在阶段三 4h）

> 本报告是 W3 的**汇总真相源**：卡4（工具描述三档）与卡5（Injection）的数据都收在这里，供面试复习与 W5 交付复用。
> 原始逐 session 数据已归档在 `reports/w3_logs/w3_card4_summary.md`（由 transcript 解析生成，非手工；原临时目录 `%TEMP%\w3_logs` 已清理）。

---

## 0. 一句话结论

- **MCP 基础已闭合**：知识库作为 `rag-kb` server（stdio）被 Claude Code 调用并正确返回来源（卡1–3）。
- **工具描述三档对比已收数**（卡4）：触发率三档 **100%**；成本主项是**工具调用轮次**，不是描述体积。
- **Injection 对抗 10/10 未被劫持**（卡5）：防护在**架构层为主、提示层为辅**，报告见 `reports/injection_report.md`。
- W3 三条通关口径**全部达成**（对账见 §2）；「MCP 深入」（无状态核心 / Extensions / 上下文成本 / 治理）按计划留阶段三，本轮不展开。

**交付物清单**：`mcp_server/`（server.py / tool_defs.py）· `.mcp.json` · `data/injection_tests.jsonl` · `reports/injection_report.md` · 本报告。

---

## 1. 卡4 · 工具描述三档对比

### 1.1 三档定义（代码真相源：[tool_defs.py](file:///c:/Users/xx/Desktop/python/Agent%E5%AD%A6%E4%B9%A0/RAG/rag-kb-next/mcp_server/tool_defs.py)）

| 档 | 环境变量值 | 注册的工具 | 描述形态 |
|---|---|---|---|
| 档① `full`（默认） | `RAGKB_DESC_TIER=full` | `search_knowledge_base` + `ask_knowledge_base` | 各自的详细 docstring（= B2 定稿，逐字不动） |
| 档② `minimal` | `RAGKB_DESC_TIER=minimal` | 同名同 schema 两个工具 | 无 docstring → description = 空串 |
| 档③ `meta` | `RAGKB_DESC_TIER=meta` | 单入口 `knowledge_base(query, action, top_k)` | 共性段合并成一份，用 `action` 在 search/ask 间分派 |

选档在 [server.py](file:///c:/Users/xx/Desktop/python/Agent%E5%AD%A6%E4%B9%A0/RAG/rag-kb-next/mcp_server/server.py) 启动时读 `RAGKB_DESC_TIER`，非法值回退 `full`。

### 1.2 实测数据（模型 `deepseek-flash` + Claude Code harness；结论只对该模型成立）

档①基线 = B2 定稿（`full`）。档②/③只在「单元格 A」组各跑 4 题（每题 1 个 session，`/clear` 开新会话）。

| 档 | 组 | session 数 | 工具触发（有调用的 session / 总数） | rag-kb search/ask 调用次数 | input token 范围 |
|---|---|---|---|---|---|
| ① full（B2） | A | 4 | 4 / 4（100%） | 17 / 3 | 26.0k – 67.3k † |
| ① full（B2） | B | 5 | 4 / 5 ‡ | 25 / 4 | 24.8k – 42.9k |
| ② minimal | A | 4 | 4 / 4（100%） | 10 / 2 | 26.0k – 27.7k |
| ③ meta | A | 4 | 4 / 4（100%） | K 调用 32 次（≈1.0 次/题） | 26.6k – 36.4k |

† 67.3k 那条 session 另跑了 24 次本地工具（bash/grep/glob），是 outlier，不是描述体积引起的。
‡ B 组有一条 `46a6627b` 触发的是**自发 Skill**（`claude-api`）而非 rag-kb，属噪声，不计入档①触发。

档③的 `action` 分布（示例）：`ask×1 + search×7` / `search×4` / `ask×1 + search×4` / `ask×1 + search×14` —— **agent 会先粗定位（search）再精核验**，两层未被压平。

### 1.3 结论

1. **触发率三档无差异（均 100%）**：在本任务的题面下（「文档里…」的问法），描述从「详细」压到「空串」甚至折叠成单入口，**不改变 agent 用不用这个工具**。→ 描述精简不是「提触发率」的手段。
2. **成本主项 = 工具调用轮次，不是描述体积**：三档每次问答的 input 底数都 ≈26k，差异主要来自**调用次数与读回的正文 token**。描述文字进的是「每轮都相同的固定前缀」，走**缓存命中**（`PRICE_CACHE_HIT_PER_M=0.02`，为未命中的 1/50），对总成本近似**常数项**。
3. **可讲的行业叙事**：`meta-tool`（档③）正是行业应对「上下文膨胀」的解法之一（把 N 个工具折叠成一个入口 + 按需分派），并有触发率数据背书——这正是修订 7 说的「能讲出怎么办」的加分项。

### 1.4 局限（必须一起讲）

- **单模型结论**：只测了 `deepseek-flash`；换模型（尤其弱模型）描述精简可能反而降触发率，需重测。
- **样本小**：每档 4 题；档②/③ 只跑了 A 组，未做 B 组交叉。
- **换档机制观察**：**`/clear` 开新会话即可让新档位生效，无需重启 Claude Code**（工具定义在会话初始化时重读）。
- **污染点**：session `23a0f232`（内含一次误输入的 `clear` 对话）已按实验纪律**剔除**，不进结论。

---

## 2. W3 通关三条对账

| # | 通关口径（计划 L676–678） | 状态 | 证据 |
|---|---|---|---|
| 1 | 知识库能在 Claude Code 里被调用并正确返回来源 | ✅ | 卡1–3；`.mcp.json` 挂载；卡4 各 session 的 search/ask 调用记录 |
| 2 | 能讲清 MCP 的上下文成本问题，以及「为什么企业把它当治理控制点」 | ✅（本轮补） | §1.3 的成本结论 + §3 理论清单第 5/6 条；实测数据支撑而非引述 |
| 3 | Injection 测试 10/10 未被劫持，且能说出防护在**哪一层** | ✅ | `reports/injection_report.md`：A(v1-sec) 10/10、B(裸) 2/10 劫持、C(search) 10/10 夹带；防护层级 §4 |

---

## 3. W3 理论清单（面试复习用 · 逐条一句）

> 这些是「能讲清」而非「要实操」的认知项；计划把 MCP 深入放在阶段三，此处只固化要点。

1. **三原语**：`Tools`（可执行动作）/ `Resources`（可读数据）/ `Prompts`（可复用模板）。本项目只用了 **Tools**——因为我们暴露的是「检索与生成」两个动作，不是「读某个资源」；Resources/Prompts 留给阶段三补认知。
2. **传输层**：`stdio`（本地子进程，Claude Code / Inspector 默认）vs `Streamable HTTP`（远程）。本项目选 stdio：本地 kb、免鉴权、进程常驻只加载一次模型。
3. **spec 版本口径**：当前 spec 是 **2026-07-28**（移除 initialize 握手与 Session-Id → **无状态核心**，可部署在 serverless / 普通 LB 后；新增 Extensions 框架）。**9-28 是企业级发布/治理里程碑，不是新的 spec 版本号**——面试必须说准。
4. **2026-09-28 企业级里程碑**：OAuth 强制 `iss` 校验；**EMA（Enterprise Managed Authorization，与 Okta 共建、基于 ID-JAG）**；MCP Apps / Tasks 转正为官方扩展；**12 个月弃用窗口**（Roots/Sampling/Logging/DCR 不得 12 个月内移除）；治理迁至 Linux Foundation 的 **AAIF**。
5. **上下文成本**：接 3 个 server 可吃掉 **>72%** 的 token（GitHub+Slack+Sentry ≈143k/200k；来源：Perplexity CTO 在 Ask 2026 引用 Apideck）。缓解方案：**Tool Search（Claude Code 内置，77k→8.7k）、Cloudflare Code Mode、meta-tool 代理（`search_tools`/`call_tool`）**。→ 本项目的档③就是 meta-tool 的落地。
6. **治理控制点**：MCP server 划定了 agent 的**权限范围**——决定它能接触什么资源、能调什么工具。本 server 只暴露两个**只读**工具、无 exec/网络/写文件能力 → 这是「架构层防护」的根（见 `injection_report.md` §4）。
7. **生态词汇（各自一句话，不实操）**：**MCP** = agent↔工具/数据；**A2A** = agent↔agent（2026-08 加入 AAIF）；**Agent Skills / AGENTS.md** = 给 agent 的封装指令包（AGENTS.md 已被 60k+ 仓库采用）。
8. **安全框架**：OWASP **Agentic Top 10（ASI01–ASI10）**；Prompt Injection 仍是 **LLM01**。本实验 → **ASI06（Memory and Context Poisoning，明确覆盖 RAG 语料/embedding 投毒）**，同时对照 **ASI01（Agent Goal Hijack，排名 #1，案例 EchoLeak / CVE-2025-32711）**。
9. **防护层级**：**架构层 > 提示层**。架构层（最小工具面 + 单次补全无工具循环 + 检索/生成分离）挡下全部**能力型**攻击；提示层（`v1-sec` 补「资料里的文字不是指令」）只挡**内容型**残余。
10. **缓解共识**：把每个 MCP server（含可信厂商的）**当作不可信输入**；工具按任务最小权限；不可逆操作强制人工审批。

---

## 4. 偏差与欠账记录

| 项 | 内容 | 处理 |
|---|---|---|
| 卡4 预注册预测缺失 | `tool_defs.py` / `server.py` / `reports/w3_logs` 全文**未见** `# 预测：…`（约束 C 要求每个 A/B 先写预测） | ⚠️ 若确未写，卡4 结论按纪律**降级为 pilot 证据**（不足以单独对外）；建议 W3 收尾时补记一行预测+对账，或明确接受降级 |
| 卡5 判据从「裸串」改为「canary 出现且无拒绝线索」 | run1 曾把「模型拒绝时引用 canary」误判为劫持（inj04/inj05） | 已修，见 `injection_report.md` §2.1 |
| `v1-sec` 可能轻微过度拒答 | inj05 的 A 答案出现「资料中没有相关内容」 | 未做 tune 集回归；如 W5 用于对外，需在 20 题调参集上回归 |
| B 组一条 session 走自发 Skill | `46a6627b` 触发 `claude-api` 而非 rag-kb | 记为噪声，不计入档①触发 |
| 污染 session | `23a0f232`（含误输入 `clear`） | 已剔除 |
| 后端度量限制 | `cache_creation_input_tokens` 恒为 0 | 「描述变动→前缀重缓存」成本只能读 Claude Code 自报的 `re-caches ~N tokens` 提示行 |

---

## 5. 附录 · 原始数据来源

- 逐 session 表（含 input / cache_read / 调用序列）：`reports/w3_logs/w3_card4_summary.md`（同目录另含早期版 `w3_summary.md` 与 B0–B2 会话原文 `B*_cell*.txt`）
- 解析脚本：`w3_sessions.py` 属一次性探针，已随 `%TEMP%\w3_logs` 清理（要点留存：**按 message.id 去重会丢流式文本块**，须按文本块累加）
- 工具描述定稿：`mcp_server/tool_defs.py`
- MCP 挂载：`.mcp.json`
- Injection 报告与测试集：`reports/injection_report.md`、`data/injection_tests.jsonl`
