"""W5 卡5 | 检索层回归门禁（需 GPU，约 1~2 分钟）。

做法：不在这里重写一遍评测，而是 subprocess 调用
    python scripts/23_eval_frozen_v2.py --check
理由：门禁与「人看的那份报告」必须是同一实现 —— 两套代码等于两个口径，
某一处改了另一处没改时，门禁会绿着而结论已经错了。

--check 比的是【严格全等】：逐题 top-k 的 chunk_id 顺序、rerank_score、汇总指标。
检索层零 API、temperature=0、RRF 有 (-rrf, chunk_id) tie-break，同一份数据重跑应逐位相同；
一旦漂了，说明代码 / 库 / 模型有实质变化，而不是「跑偏了一点」。

运行：
    & "<venv>\\Scripts\\python.exe" -m pytest tests/test_retrieval_baseline.py -q
    （只跑快测试：-m "not slow"）
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE_SCRIPT = ROOT / "scripts" / "23_eval_frozen_v2.py"


@pytest.mark.slow
def test_frozen_v2_retrieval_matches_baseline():
    assert BASELINE_SCRIPT.exists(), "评测脚本缺失"
    proc = subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT), "--check"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-20:])
    assert proc.returncode == 0, f"检索层回归失败（exit={proc.returncode}）：\n{tail}"
    assert "回归通过" in proc.stdout, f"输出里没有回归通过标记：\n{tail}"
