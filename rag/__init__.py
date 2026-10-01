"""RAG 知识库问答包 | Day9 从 scripts/ 各日脚本抽取，作为 Day10+ 的唯一底座。

分层（自上而下依赖，反向依赖一律禁止）：
    cli / eval        入口层：交互问答、批量评测
      └─ pipeline     编排层：召回 → 精排 → 去重 → 生成，对外只有一个 answer()
           ├─ retriever   检索层：向量召回（bi-encoder）+ 精排（cross-encoder）+ 去重
           ├─ llm         生成层：DeepSeek 调用、prompt 拼装、token/成本
           ├─ judge       裁判层：LLM-as-judge 两指标（答案正确率 / 引用正确率）
           └─ ingest      入库层：chunks.jsonl → Chroma
                └─ embedding 嵌入层：全项目唯一的 BGE 封装（别处禁止再写一份）
                     └─ config 配置层：路径 / 模型 / 参数 / prompt 常量

对照：scripts/01–08 是「重构前」的历史脚本，一行不改，保留可复现的对照数据。
"""