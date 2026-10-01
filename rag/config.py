"""配置层 | Day9：全项目唯一的「参数 / 常量」出口。

为什么要有它：Day1–8 的脚本各自 hardcode 了 DB_PATH / COLLECTION_NAME / prompt 模板 /
检索参数，改一个值要翻 8 个文件、还容易漏改（Day6 的 collection 名、Day8 的 top_k 各写一份）。
收进 config 后，其余模块只 import 常量，字面量不再出现在业务代码里。
"""
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent      # 项目根 rag-kb/
DATA_DIR = BASE_DIR / "data"
CHUNKS_JSONL = DATA_DIR / "processed" / "chunks.jsonl"
TEST_JSONL = DATA_DIR / "test_questions.jsonl"
# Day10 卡 4 起换成 prompt 专项结果文件：eval_day10.json 留在磁盘作「卡2/3 配额 A/B」的存档
EVAL_JSON = DATA_DIR / "eval_day10_prompt.json"
DB_PATH = BASE_DIR / "chroma_db"
WEB_DIR = BASE_DIR / "web"      # Day13 前端静态目录，由 api/main.py 挂成同源（见文末 mount）

# ---------------- 可观测：本地自研 trace（Day10 卡 5，零账号方案） ----------------
# 落盘形态：单文件 append-only —— 一次问答一行 / 一条反馈一行
TRACES_DIR = DATA_DIR / "traces"
TRACES_JSONL = TRACES_DIR / "traces.jsonl"
SCORES_JSONL = TRACES_DIR / "scores.jsonl"

COLLECTION_NAME = "day6_kb"     # 正式库（Day14 交付时 932 块）；day2_kb 是 Day1–2 玩具库，别碰
BATCH_SIZE = 64                 # 入库分批大小
DEFAULT_K = 5                   # 未精排时的默认召回数（Day6/07 的口径）

# ---------------- 上传与切块（Day12 卡 3） ----------------
UPLOAD_DIR = DATA_DIR / "uploads"   # 上传的原始 PDF 落盘处（chunks.jsonl 之外的第二个数据来源）
CHUNK_SIZE = 300                # 932 块库的口径（实测 max char_count=330 = 300 + 10% overlap）。
                                # 上传必须用同一个值：口径不同 → 块长分布不同 → 精排/去重表现不可比
UPLOAD_MAX_MB = 20              # 上传大小上限（避免一次请求吃掉内存与显存）

# 本地模型：bge-small-zh 是 9 月已下（嵌入），bge-reranker-base 是 Day8 新下（精排）
EMBED_MODEL_PATH = r"c:\Users\xx\Desktop\python\Agent学习\PyTorch\HuggingFace\models\bge-small-zh"
RERANKER_MODEL_PATH = r"c:\Users\xx\Desktop\python\Agent学习\PyTorch\HuggingFace\models\bge-reranker-base"

# LLM：deepseek-v4-flash。注意 v4 默认开思考模式——temperature/top_p 不报错但不生效，
# 思维链走 reasoning_content 字段；普通问答用 extra_body 关掉它（Day6 官网核实的坑）
LLM_MODEL = "deepseek-v4-flash"
THINKING_DISABLED = {"thinking": {"type": "disabled"}}
# Day10 改动：0.7 → 0。理由＝Day9 通关题 K11 的结论——温度 0.7 时 A/B 两组有 6 题
# top3 完全相同、指标却不同，说明采样噪声盖过了实验变量。做实验要先锁死随机源。
TEMPERATURE = 0

# 计价（元 / 百万 token），Day9 起把成本算进基线表（缓存命中价低得多，这里按未命中估）
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 2.0

# ---------------- 检索参数（Day9 A/B 实验的变量都在这三行） ----------------
RECALL_K = 10           # 第一阶段：向量候选池深度（要大于精排数，才有得排）
RERANK_TOP_K = 3        # 第二阶段：送进 prompt 的条数（Day8 现状）
CROSS_TOP_K = 5         # Day9 已否：cross 单独放宽 top_k 是纯负收益。
                        # Day14 起【默认问答路径不再读它】（pipeline._retrieve 统一用 RERANK_TOP_K），
                        # 只剩 eval.retrieval_main 的检索层对照按题型取它 —— 留着是为了让 Day9 那组对照可复现。
CROSS_TYPES = {"cross"}
SOURCE_CAP = 2          # Day10 新增：同一 source 在 top-k 里最多占几条（治同源垄断）

# ---------------- Prompt 模板（Day7 版：真实编号 [k] + 有条件拒答） ----------------
# Day10 卡 4：拒答门槛实验。所有 system prompt 都收进 PROMPTS、由 pipeline 按版本取用，
SYSTEM_PROMPT_V1 = (
    "你是一个基于个人知识库的问答助手，只能依据用户提供的资料回答问题。\n"
    "回答时，凡是结论、步骤、解释等来自某个【资料k】的内容，都要在该句后标注[k]；"
    "k 必须是对应资料的真实编号（1~5），例如由【资料2】支撑的句子就标 [2]，"
    "绝不要写成 [n]、[k] 这类占位符。如需用到多个资料，就分别标注各自编号。\n"
    "如果资料与问题完全无关、确实无法支撑任何回答，才明确说\"资料中没有相关内容\"，绝不编造；"
    "只要资料包含相关的概念、原理或步骤，就应基于资料作答并标注引用。\n"
    "用中文，简洁准确。"
)

# system_prompt--「拆子问题 + 部分作答」版，用来压低整题拒答率。
SYSTEM_PROMPT_V2 = (
    "你是一个基于个人知识库的问答助手，只能依据用户提供的资料回答问题。\n"
    "先给出问题包含的子问题清单，再逐条回答\n"
    "若子问题有资料支撑，回答并标注资料真实编号\n"
    "回答时，凡是结论、步骤、解释等来自某个【资料k】的内容，都要在该句后标注[k]；"
    "k 必须是对应资料的真实编号（1~5），例如由【资料2】支撑的句子就标 [2]，"
    "绝不要写成 [n]、[k] 这类占位符。如需用到多个资料，就分别标注各自编号。\n"
    "若子问题没有资料支撑，回答「资料未涉及……」，禁止编造内容\n"
    "不得因为「某部分资料没覆盖」就把整题判为无关而拒绝回答\n"
    "用中文，简洁准确。"
)

PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}
USER_TEMPLATE = "资料：\n{context}\n\n问题：{question}"

EVAL_FINAL_JSON = DATA_DIR / "eval_day14.json"   # Day14：交付基线（单组 final 口径）


# 按版本取 system prompt，取不到就快速失败
def resolve_prompt(version: str) -> str:
    prompt = PROMPTS.get(version)
    if not prompt:
        raise ValueError(f"config.PROMPTS[{version!r}] 不存在（现有版本：{sorted(PROMPTS)}）")
    return prompt


load_dotenv(str(BASE_DIR / ".env"))   # 显式路径，不依赖当前工作目录