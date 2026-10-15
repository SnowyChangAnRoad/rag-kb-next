import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 用当前 venv 的 python 解释器启动 server 子进程。
    # 显式给 cwd：否则 args 里的相对路径 / `-m` 的导入根都取决于「父进程在哪跑」——
    # 卡3 的 .mcp.json 会走同一条路，这里先把这个坑钉死。
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 打印 tools/list 结果
            tools = await session.list_tools()
            print("Tools:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            # 调用
            search_result1 = await session.call_tool(
                "search_knowledge_base", arguments={"query": "Day6 用了什么嵌入模型", "top_k": 3})
            print(f"search_knowledge_base() = {search_result1.structured_content}")
            print(f"search_content  = {search_result1.content}")

            search_result2 = await session.call_tool(
                "search_knowledge_base", arguments={"query": "今天天气怎么样", "top_k": 3})
            print(f"search_knowledge_base() = {search_result2.structured_content}")
            print(f"search_content  = {search_result2.content}")

            search_result3 = await session.call_tool(
                "search_knowledge_base", arguments={"query": "Redis是什么", "top_k": 999})
            print(f"search_knowledge_base() = {search_result3.structured_content}")
            print(f"search_content  = {search_result3.content}")

            ask_result = await session.call_tool(
                "ask_knowledge_base", arguments={"question": "知识库里讲了什么检索模式"})
            print(f"ask_knowledge_base() = {ask_result.structured_content}")
            print(f"ask_content  = {ask_result.content}")

if __name__ == "__main__":
    asyncio.run(main())