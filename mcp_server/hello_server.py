from mcp.server import MCPServer

mcp = MCPServer("rag-kb-demo")

# 工具描述(docstring) "Add two numbers" 会被模型读取
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()