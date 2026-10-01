"""
Day 1 | 环境连通性测试：.env 读取 + DeepSeek API 一次最小调用
用法：python scripts/day1_env_check.py（从任意目录运行均可，.env 按脚本位置定位）
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 不依赖当前工作目录：从 scripts/ 的上一级（项目根）找 .env
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)
print(f".env 位置：{ENV_PATH}  存在：{ENV_PATH.exists()}")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key or api_key == "sk-your-key-here":
    raise SystemExit("DEEPSEEK_API_KEY 未配置：请把真实 key 填入项目根目录的 .env")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "用一句话回答：RAG 的检索器和生成器各负责什么？"}
    ],
    max_tokens=100,
)
print("DeepSeek 回答：", resp.choices[0].message.content)
print("用量 tokens：", resp.usage.prompt_tokens, "+", resp.usage.completion_tokens)
print("ENV CHECK OK")
