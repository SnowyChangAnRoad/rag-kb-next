"""兼容壳 | Day9：实现已迁到 rag/embedding.py，这里只做 re-export。

为什么保留这个文件而不是直接删：
  01/02/06/07/08 里都有 `from common_embedding import BGEEmbeddingFunction`，
  删掉会让这批历史脚本报错。留一个 3 行兼容层，旧脚本一行不改，
  同时「全项目只允许有一份嵌入实现」的硬规则也保住了——真代码只有一份。

这行 sys.path 是为了让 scripts/ 下的脚本运行时能找到项目根下的 rag 包。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedding import BGEEmbeddingFunction  # noqa: E402,F401