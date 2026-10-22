"""要求 7：批量评估 —— 跑 eval/questions.json 中 20 条测试问题做流程测试：每条问题走一遍
完整 Agent 链路（07 的 ask：模型决策 → 工具调用 → 回答 → 06 轨迹落盘），只记录流程表现，
不验证答案内容。流程判定仅看：ask 正常返回且回答非空即通过（单条异常不中断整体）。

输出：逐条回答、本轮工具调用与耗时写入 eval/eval_report.json，控制台打印流程摘要
（通过率、工具调用分布、平均耗时与失败明细），回答全文保留在报告中供人工分析。

运行方式：python 08_evaluation.py [limit]，limit 可选（只跑前 N 条，用于快速冒烟）。
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

base = Path(__file__).parent

# 复用 07 的模块加载器（agent._sib 内部有缓存，避免 bge 模型重复加载）
_spec = importlib.util.spec_from_file_location("rag_agent_07", str(base / "07_rag_agent.py"))
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)  # 07 在导入时即加载 01-06（含 load_dotenv）

TRACE_FILE = agent._sib("06_trace_logging").TRACE_FILE
QUESTIONS_FILE = base / "eval" / "questions.json"
REPORT_FILE = base / "eval" / "eval_report.json"


def _count_tools_used(trace_offset: int) -> list[str]:
    """读取 traces.jsonl 自 trace_offset 字节之后的新增记录，返回本轮调用的工具名列表。"""
    if not TRACE_FILE.exists():
        return []
    # 按字节切片（offset 来自 st_size）：文件含中文，若按字符切会与字节数错位
    text = TRACE_FILE.read_bytes()[trace_offset:].decode("utf-8", errors="replace")
    tools = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("type") == "tool_call":
            tools.append(record.get("tool"))
    return tools


def run_one(question: str, client) -> dict:
    """对单条问题做流程测试：调用 07 的 ask()，记录回答、工具调用与耗时，不验证答案。"""
    offset = TRACE_FILE.stat().st_size if TRACE_FILE.exists() else 0
    t0 = time.time()
    try:
        answer = agent.ask(question, client)
        error = None
    except Exception as e:  # 单条失败不中断整体评估
        answer, error = "", f"{type(e).__name__}: {e}"
    entry = {
        "question": question,
        "answer": answer,
        "tools_used": _count_tools_used(offset),
        "elapsed_s": round(time.time() - t0, 2),
        "pass": bool(answer.strip()),  # 流程判定：正常返回且回答非空
    }
    if error:
        entry["error"] = error
    return entry


def build_report(results: list[dict]) -> dict:
    """汇总流程指标：通过率、工具调用分布与失败明细。"""
    tool_calls = {}  # 工具名 -> 调用次数
    for r in results:
        for name in r["tools_used"]:
            tool_calls[name] = tool_calls.get(name, 0) + 1
    passed = sum(bool(r["pass"]) for r in results)
    elapsed = [r["elapsed_s"] for r in results]
    failures = [
        {"question": r["question"], "error": r.get("error", "回答为空")}
        for r in results if not r["pass"]
    ]
    return {
        "meta": {
            "total": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4) if results else None,
            "avg_elapsed_s": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
        },
        "tool_call_stats": {
            "total_calls": sum(tool_calls.values()),
            "by_tool": tool_calls,
            "questions_using_tools": sum(bool(r["tools_used"]) for r in results),
        },
        "failures": failures,
        "details": results,
    }


def run(limit: int | None = None) -> dict:
    """批量流程测试主入口：跑全部（或前 limit 条）问题，写报告文件并返回报告。"""
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    if limit:
        questions = questions[:limit]
    client = agent.get_client()

    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        entry = run_one(question, client)
        results.append(entry)
        note = f"（error: {entry['error']}）" if entry.get("error") else ""
        print(f"    -> {'OK' if entry['pass'] else 'FAIL'}{note}，"
              f"工具调用：{entry['tools_used'] or '无'}，耗时 {entry['elapsed_s']}s")

    report = build_report(results)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m, ts = report["meta"], report["tool_call_stats"]
    print(f"\n===== 流程测试摘要 =====")
    print(f"总体：{m['passed']}/{m['total']} 通过（通过率 {m['pass_rate']}），平均耗时 {m['avg_elapsed_s']}s")
    print(f"工具调用：共 {ts['total_calls']} 次（{ts['by_tool']}），"
          f"其中 {ts['questions_using_tools']} 条问题触发了工具调用")
    if report["failures"]:
        print(f"\n----- 失败明细（{len(report['failures'])} 条）-----")
        for f in report["failures"]:
            print(f"    - 「{f['question']}」：{f['error']}")
    print(f"\n报告已写入：{REPORT_FILE}")
    return report


def main() -> None:
    """以函数调用方式运行：命令行可选 limit 参数，只跑前 N 条用于快速冒烟。"""
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    run(limit=limit)


if __name__ == "__main__":
    main()
