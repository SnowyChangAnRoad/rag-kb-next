"""验收脚本 | Day14：把阶段总验收 6 条变成一键可复现的证据。

为什么要有它（而不是截图）：「我看过了，能跑」换台机器就没人知道当时验的是什么；
脚本化之后这张验收表能直接进 README，面试也能当场跑。

前置：另开终端 uvicorn api.main:app --port 8000
"""
import json, subprocess, sys, time, unicodedata
import re
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
EVAL_FINAL = ROOT / "data" / "eval_day14.json"
TRACES = ROOT / "data" / "traces" / "traces.jsonl"
SCORES = ROOT / "data" / "traces" / "scores.jsonl"

IN_QUESTION = "RAG 有哪些局限性？"                 # 库内题（答案在 RAG知识.pdf）
# 库外题：必须和库里 4 份资料的任何一块都无关，否则测不出「拒答」。
# 注意这里不能留空串 —— ChatRequest.question 有 min_length=1，空串直接 400，
# 而 400 会让 probe_stream 抛 HTTPStatusError，看起来像「脚本坏了」，其实是参数没填。
OUT_QUESTIONS = [
    "我家的洗衣机显示 E4 错误码，该怎么修？",
    "2026 年世界杯的冠军是哪支球队？",
]

def parse_frame(lines: list[str]) -> dict | None:
    # TODO-2a：把一帧的若干行（event: x / data: {...}）解析成 {"event":..., "data": {...}}。
    event = None
    data_lines: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    if event is None or not data_lines:
        return None

    try:
        data = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None

    return {"event": event, "data": data}

def probe_stream(question: str, qtype: str = "in") -> dict:
    """TODO-2b：流式探针 —— 实测首 token 延迟（TTFT）+ 校验帧顺序。

    【本卡的核心考点】curl 的 `time_starttransfer` 测不到 TTFT：
    它记的是【响应头】到达的时间，而 SSE 的响应头在生成器第一次 yield 之前就发出去了
    （与 Day12「参数校验必须在建流前」同一条原理）。首 token 只能读到流里第一个 delta 才知道。

    返回 {"ttft_ms": float, "order_ok": bool, "sources": [...], "answer": str}
    """
    t0 = time.perf_counter()
    ttft_ms = None            # 第一个 delta 到达的时刻（而不是响应头到达的时刻）
    src_idx = None            # sources 帧是第几帧
    delta_idx = None          # 第一个 delta 帧是第几帧
    sources: list[dict] = []
    pieces: list[str] = []
    buf: list[str] = []       # 正在攒的一帧
    idx = -1                  # 帧计数器（从 0 开始，用于给出「sources #1 / delta #2」这种硬证据）

    # 【关键】必须是 httpx.stream 而不是 httpx.post：
    # httpx.post 会把整个响应体读完才返回 → 那时测到的是「整条流时长」，不是首 token。
    # 这与 Day12 后端「本层不攒内容」是同一条纪律，只是换到了客户端一侧。
    with httpx.stream("POST", f"{BASE}/chat/stream",
                      json={"question": question, "qtype": qtype},
                      timeout=60.0) as r:                 # 生成要 1~2s，超时别用默认 5s
        r.raise_for_status()
        # iter_lines() 是【增量】产出：每来一行就交出，绝不等到流结束 —— TTFT 的前提。
        # 它已经把行尾 \n 去掉，所以空行就是 ""。
        for line in r.iter_lines():
            if line != "":
                buf.append(line)                          # 字段行，先攒着
                continue
            # 走到这里 = 一个空行 = 一帧结束（Day12 手拼帧时那个「结尾空行」的对面）
            idx += 1
            frame = parse_frame(buf)                      # 复用你写的解析器
            buf = []
            if not frame:                                 # 坏帧/空帧：跳过而不是崩
                continue
            ev, data = frame["event"], frame["data"]
            if ev == "sources":
                src_idx = idx
                sources = data["sources"]
            elif ev == "delta":
                if delta_idx is None:                     # 只在【第一个】delta 上取时刻
                    delta_idx = idx
                    ttft_ms = (time.perf_counter() - t0) * 1000
                pieces.append(data["text"])
            # meta / done 帧这里用不上，显式忽略即可（不写 else，避免误吞未知事件名）

    # 契约断言：sources 必须排在第一个 delta 之前（Day13 定的顺序契约，今天把它自动化）
    order_ok = (src_idx is not None and delta_idx is not None and src_idx < delta_idx)
    return {"ttft_ms": ttft_ms, "order_ok": order_ok,
            "sources": sources, "answer": "".join(pieces)}

def is_refusal(answer: str) -> bool:
    """TODO-3：判定「明确拒答」。写出你的判据，并说明为什么不能只查一个关键词。"""
    text = (answer or "").strip()
    if not text:
        return False

    refusal_markers = ("未涉及","未提及","没有相关","未找到相关","无法回答","无法根据","不能回答","没有足够信息","抱歉")

    has_refusal = any(marker in text for marker in refusal_markers)
    has_citation = re.search(r"\[\d+\]", text) is not None

    return has_refusal and not has_citation

def _width(s: str) -> int:
    """按【显示宽度】算长度：CJK 汉字与全角标点占 2 列。

    等宽字体下直接用 len() 排中文表格必然错位（'库' 的 len 是 1、占 2 列）。
    这是排版问题，不是编码问题 —— 但中文项目里天天撞。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _width(s))


def _read_jsonl(path: Path) -> list[dict]:
    """逐行 json 解析；文件不存在返回空表（让调用方去判 FAIL，而不是在这里抛异常）。"""
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _check_requirements() -> tuple[str, str, str]:
    """requirements.txt 钉的版本 vs 实际装上的版本。

    三条必须说清的规则，否则会假报不匹配：
      ① 包名大小写不敏感（PyPI 规范），统一 lower；
      ② `-` 与 `_` 等价（python-multipart == python_multipart）；
      ③ requirements.txt 里【没列】的包不在比对范围 —— 但那正是「可复现缺口」本身：
         别人照这份文件装不出能跑的环境，所以单独把它报出来，不能装作没看见。
    """
    pinned: dict[str, str] = {}
    for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()          # 去掉行尾注释，否则 "chromadb==1.5.5  # 注释" 解析不出来
        if "==" in ln:
            name, ver = ln.split("==", 1)
            pinned[name.strip().lower().replace("_", "-")] = ver.strip()

    # 用 sys.executable 而不是裸 "pip"：保证查的是【当前这个解释器】的包，
    # 否则在 venv 里跑脚本、却去问系统 pip，比对结果全是错的。
    p = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    installed: dict[str, str] = {}
    for ln in p.stdout.splitlines():
        if "==" in ln:
            name, ver = ln.split("==", 1)
            installed[name.strip().lower().replace("_", "-")] = ver.strip()

    bad = [f"{n}: 钉 {v} ≠ 实装 {installed.get(n, '未安装')}"
           for n, v in pinned.items() if installed.get(n) != v]
    unlisted = [n for n in ("torch", "transformers", "httpx") if n in installed and n not in pinned]

    ev = f"{len(pinned)} 个钉版本"
    ev += "，全部一致" if not bad else "；不一致：" + " | ".join(bad)
    if unlisted:
        ev += f"；未列出但实装：{'/'.join(unlisted)}（可复现缺口）"
    return ("工程规范·requirements 版本可复现", "PASS" if not bad else "FAIL", ev)


def check_static() -> list[tuple[str, str, str]]:
    """读文件类检查：验收 #3（可观测）与 #6（工程规范）。

    【为什么不 import rag.*】验收脚本若调用被测代码的内部函数，被测代码错了它也会跟着错，
    变成自证循环。直接读 jsonl 原文 = 更独立的第二读者，与 parse_frame 的价值同理。
    """
    rows: list[tuple[str, str, str]] = []

    # ---------- 验收 #3：可观测 ----------
    traces = _read_jsonl(TRACES)
    if not traces:
        rows.append(("可观测·trace 完整", "FAIL", f"{TRACES.name} 不存在或为空"))
    else:
        rec = traces[-1]
        names = {s["name"] for s in rec.get("spans", [])}
        cost = rec.get("totals", {}).get("cost", 0) or 0
        ok = {"retrieval", "generation"} <= names and cost > 0
        rows.append(("可观测·trace 含检索+生成 span 与成本", "PASS" if ok else "FAIL",
                     f"{rec['trace_id']} spans={sorted(names)} 成本 ¥{cost:.4f}"))

    # 反馈按 trace_id 做 join：能 join 上的才是「有效反馈」，join 不上的叫孤儿。
    # 分母用【有效】数而不是总条数 —— 否则写条假记录就能把验收刷过去。
    scores = _read_jsonl(SCORES)
    known = {r["trace_id"] for r in traces}
    valid = [s for s in scores if s["trace_id"] in known]
    orphan = [s for s in scores if s["trace_id"] not in known]
    up = sum(1 for s in valid if s["value"] == "up")
    rows.append(("可观测·有效用户反馈 ≥3", "PASS" if len(valid) >= 3 else "FAIL",
                 f"有效 {len(valid)} 条（👍{up} / 👎{len(valid) - up}），"
                 f"孤儿 {len(orphan)} 条"
                 + (f" {[s['trace_id'] for s in orphan]}" if orphan else "")))

    # ---------- 验收 #6：工程规范 ----------
    has_env = (ROOT / ".env").exists() and (ROOT / ".env.example").exists()
    rows.append((".env 存在且有 .env.example", "PASS" if has_env else "FAIL",
                 ".env + .env.example 均在项目根（前者不入库、后者给模板）"))

    ignore = ROOT / ".gitignore"
    lines = [ln.strip() for ln in ignore.read_text(encoding="utf-8").splitlines()] if ignore.exists() else []
    rows.append((".gitignore 忽略 .env", "PASS" if ".env" in lines else "FAIL",
                 '按【整行】匹配：用子串 ".env" in text 会被 .env.example 误判成已忽略'))

    # git 还没 init 时 returncode≠0 —— 报 PENDING 而不是崩，也不假装通过
    p = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        rows.append((".env 未被 git 跟踪", "PENDING", "尚无 git 仓库（卡5 处理）"))
    else:
        tracked = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        rows.append((".env 未被 git 跟踪", "FAIL" if ".env" in tracked else "PASS",
                     f"git ls-files 共 {len(tracked)} 个文件，其中无 .env"))

    rows.append(_check_requirements())
    return rows


def main() -> None:
    """验收 #1~#6 全跑一遍，打一张表。

    结果取值只有四种：PASS / FAIL / PASS(初判) / MANUAL / PENDING。
    故意不用「✓✗」两态 —— 把「机器只能初判」和「只能人判」的部分如实标出来，
    比一张漂亮的全绿表可信得多。
    """
    rows: list[tuple[str, str, str]] = []
    manual: list[str] = []

    # ---------- 前置：服务活着吗 ----------
    try:
        health = httpx.get(f"{BASE}/health", timeout=10.0)
    except httpx.ConnectError:
        print(f"连不上 {BASE}。先在另一个终端起服务：uvicorn api.main:app --port 8000")
        raise SystemExit(1)

    # ---------- 验收 #4：API 可用 ----------
    docs = httpx.get(f"{BASE}/docs", timeout=10.0)
    rows.append(("4 API·/docs 可交互", "PASS" if docs.status_code == 200 else "FAIL",
                 f"HTTP {docs.status_code}"))
    h = health.json()
    rows.append(("4 API·/health 200", "PASS" if health.status_code == 200 else "FAIL",
                 f"status={h.get('status')} collection={h.get('collection')} "
                 f"chunks={h.get('chunks')} default_top_k={h.get('default_top_k')}"))

    # 库 == 真相源：Day13 那条判据，今天变成验收表的一行（走 /health，不必 import chroma）
    n_lines = len(_read_jsonl(ROOT / "data" / "processed" / "chunks.jsonl"))
    rows.append(("前置·库内容 == chunks.jsonl", "PASS" if h.get("chunks") == n_lines else "FAIL",
                 f"collection {h.get('chunks')} vs 真相源 {n_lines} 行"))

    # ---------- 验收 #1：流式 + 顺序契约 + 引用 ----------
    # 【为什么要连打两枪】首 token 延迟有两个测量条件，混在一起说等于没说：
    #   · 冷启动 = 服务起来后的第一个请求，含 CUDA kernel 首次编译、精排模型首次推理、
    #     到 DeepSeek 的 TLS 握手 —— 会稳定地把 2s 验收线顶穿；
    #   · 稳态 = 之后每个请求的真实体感。
    # 判定用稳态，但冷启动值必须一起报出来：只报好看的那个数字，就是自欺。
    # 这里【故意不写具体毫秒】：脚本每次跑都会把两个值打进验收表（见 README 验收一节），
    # 注释里写死必然随时间过期 —— 本行就曾长期留着一对与当时实测不符的旧值。
    cold = probe_stream(IN_QUESTION, "in")
    st = probe_stream(IN_QUESTION, "in")
    ttft = st["ttft_ms"] or -1          # 没收到 delta 时是 None，先归一成负数再比，避免 f-string 崩
    cold_ms = cold["ttft_ms"] or -1
    rows.append(("1 端到端·首 token <2s（稳态）", "PASS" if 0 < ttft < 2000 else "FAIL",
                 f"稳态 {ttft:.0f}ms / 冷启动 {cold_ms:.0f}ms（均含检索，从发出请求算起）"
                 if ttft > 0 else "未收到任何 delta 帧"))
    rows.append(("1 端到端·sources 先于 delta", "PASS" if st["order_ok"] else "FAIL",
                 "Day13 定的顺序契约，今天第一次被自动断言"))
    has_cite = re.search(r"\[\d+\]", st["answer"]) is not None
    rows.append(("1 端到端·回答带来源角标 [n]", "PASS" if has_cite else "FAIL",
                 f"{len(st['sources'])} 个来源 / 回答 {len(st['answer'])} 字"))

    # ---------- 验收 #1b：库外拒答（机器初判 + 人工复核）----------
    n_ok = 0
    for q in OUT_QUESTIONS:
        r = probe_stream(q, "out")
        ok = is_refusal(r["answer"])
        n_ok += ok
        manual.append(f"【需人工复核】库外题「{q}」\n"
                      f"  机器初判：{'拒答' if ok else '未拒答'}（判据=含拒答语 且 无 [n]）\n"
                      f"  原文：{r['answer'][:400]}")
    rows.append((f"1 端到端·库外拒答 {n_ok}/{len(OUT_QUESTIONS)}", "PASS(初判)" if n_ok == len(OUT_QUESTIONS) else "FAIL",
                 "自动判据只能证明「没说带引用的话」，编造与否见人工复核区"))

    # ---------- 验收 #2：效果基线（读 eval_day14.json）----------
    if EVAL_FINAL.exists():
        s = json.loads(EVAL_FINAL.read_text(encoding="utf-8"))
        sm, lib = s["summary"], s["library"]
        ok = sm["answer_acc"] >= 0.8 and sm["cite_acc"] >= 0.8
        rows.append(("2 效果基线·答案/引用 ≥80%", "PASS" if ok else "FAIL",
                     f"acc {sm['answer_acc'] * 100:.1f}% / cite {sm['cite_acc'] * 100:.1f}% / "
                     f"覆盖 {sm['cite_at_k'] * 100:.1f}%（{lib['collection']} @ {lib['count']} 块）"))
    else:
        rows.append(("2 效果基线·答案/引用 ≥80%", "FAIL", "eval_day14.json 不存在：先跑 python -m rag.eval final"))

    # ---------- 验收 #3 / #6 ----------
    rows.extend(check_static())

    # ---------- 验收 #5：只能人判 ----------
    rows.append(("5 概念口述 5 题", "MANUAL", "见 README「面试可讲的 5 个问题」"))

    # ---------- 打表 ----------
    print("\n" + "=" * 100)
    print(f"阶段总验收  {time.strftime('%Y-%m-%d %H:%M:%S')}  服务 {BASE}")
    print("=" * 100)
    print(_pad("验收项", 42) + _pad("结果", 14) + "证据")
    print("-" * 100)
    for item, status, ev in rows:
        print(_pad(item, 42) + _pad(status, 14) + ev)
    print("-" * 100)
    n_pass = sum(1 for _, s, _ in rows if s.startswith("PASS"))
    n_fail = sum(1 for _, s, _ in rows if s == "FAIL")
    n_other = len(rows) - n_pass - n_fail
    print(f"PASS {n_pass} / FAIL {n_fail} / 人工+待办 {n_other}")

    if manual:
        print("\n" + "=" * 100)
        print("人工复核区（自动判据只当筛子，不当法官）")
        print("=" * 100)
        for m in manual:
            print(m + "\n")


if __name__ == "__main__":
    main()