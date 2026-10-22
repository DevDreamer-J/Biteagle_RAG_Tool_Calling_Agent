"""要求 6：轨迹日志 —— 以装饰器方式为 Agent 记录每一步的 reasoning 摘要、工具调用和最终结果，
按行追加写入 data/traces.jsonl（JSON Lines，UTF-8，中文不转义）。

用法（07 Agent 中）：
    @trace_agent
    def run_agent(question: str, tracer: AgentTracer) -> str:
        tracer.log_reasoning("问题需要查知识库，先检索")
        result = dispatch_tool("query_pdf_files")
        tracer.log_tool("query_pdf_files", {}, result)
        ...
        return answer  # 返回值被装饰器自动记录为 final_result

行为约定：
  - 进入被装饰函数：创建 tracer（生成 trace_id），记录 start 事件；
  - 函数体内随时调用 tracer.log_reasoning / log_tool / log_info 记录各步骤（每步立即落盘）；
  - 正常返回：自动记录 final_result（answer=返回值、耗时、status=success）；
  - 抛出异常：先记录 final_result（status=error、error 信息）再原样抛出，不吞异常。

实现逻辑写在函数/类内，文件末尾以 __main__ 入口直接调用函数演示（python 06_trace_logging.py），便于 tests 导入。
"""

import functools
import inspect
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# 轨迹日志文件（相对本文件，从任何工作目录运行都能找到）
TRACE_FILE = Path(__file__).parent / "data" / "traces.jsonl"


def _now_iso() -> str:
    """当前时间的 ISO 格式字符串（精确到秒）。"""
    return datetime.now().isoformat(timespec="seconds")


class AgentTracer:
    """单次 Agent 运行的轨迹记录器（由 trace_agent 装饰器创建并注入，业务代码直接调用 log_* 方法）。

    每次调用 log_* 都立即追加一行 JSON 到 traces.jsonl，中途崩溃时已记录的步骤不丢失；
    同一次运行的所有行共享同一个 trace_id，便于按会话回放。
    """

    def __init__(self, trace_id: str, log_file: Path, t0: float | None = None) -> None:
        self.trace_id = trace_id
        self._log_file = log_file
        self._step = 0  # 步骤号，从 1 递增
        self._t0 = t0 if t0 is not None else time.time()  # 起点，用于 final_result 耗时统计

    def _write(self, type_: str, payload: dict[str, Any]) -> None:
        """组装一行记录（trace_id + 时间戳 + 递增步骤号 + 类型 + 载荷）并立即落盘。"""
        self._step += 1
        record = {"trace_id": self.trace_id, "ts": _now_iso(), "step": self._step, "type": type_, **payload}
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_reasoning(self, summary: str, **extra: Any) -> None:
        """记录一步 reasoning 摘要（Agent 的决策/思考说明，summary 一句话即可）。"""
        self._write("reasoning", {"summary": summary, **extra})

    def log_tool(self, name: str, arguments: Any, result: Any, **extra: Any) -> None:
        """记录一次工具调用（工具名、入参、返回结果）。"""
        self._write("tool_call", {"tool": name, "arguments": arguments, "result": result, **extra})

    def log_info(self, event: str, **extra: Any) -> None:
        """记录其他通用事件（如：决定拒答、检索为空、引用来源等）。"""
        self._write("info", {"event": event, **extra})

    def log_final(self, answer: Any, status: str = "success", **extra: Any) -> None:
        """记录最终结果（装饰器在函数返回/异常时自动调用，业务代码一般无需手动调用）。"""
        self._write(
            "final_result",
            {"answer": answer, "status": status, "elapsed_s": round(time.time() - self._t0, 3), **extra},
        )


def trace_agent(func: Callable | None = None, *, log_file: Path = TRACE_FILE):
    """装饰器：为 Agent 主函数注入 AgentTracer 并自动记录 start / final_result。

    - 被装饰函数需在签名中声明 tracer 参数（如 def run(question, tracer)），
      装饰器以关键字参数 tracer=<AgentTracer> 注入；签名中没有 tracer 参数时报错。
    - @trace_agent 与 @trace_agent(log_file=...) 两种写法均可。
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        if "tracer" not in sig.parameters:
            raise TypeError(f"{fn.__name__} 的签名中必须包含 tracer 参数以接收 AgentTracer")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            tracer = AgentTracer(trace_id=uuid.uuid4().hex[:12], log_file=log_file, t0=t0)
            tracer.log_info("agent_start", function=fn.__name__)
            kwargs["tracer"] = kwargs.get("tracer") or tracer
            try:
                answer = fn(*args, **kwargs)
            except Exception as e:
                tracer.log_final(answer=None, status="error", error=f"{type(e).__name__}: {e}")
                raise  # 不吞异常：先落盘失败记录再原样抛出
            tracer.log_final(answer=answer, status="success")
            return answer

        return wrapper

    return decorator(func) if func is not None else decorator


def main() -> None:
    """以函数调用方式演示：装饰一个模拟 Agent，观察 traces.jsonl 的记录效果。"""
    @trace_agent
    def demo_agent(question: str, tracer: AgentTracer) -> str:
        tracer.log_reasoning("问题涉及当前时间，决定调用时间工具")
        tool_result = query_time_stub()
        tracer.log_tool("query_current_time", {}, tool_result)
        return f"现在是 {tool_result}，已回答问题：{question}"

    def query_time_stub() -> str:
        return "2026-10-22 16:00:00"

    print("=== 演示：装饰器记录 Agent 轨迹 ===")
    print(demo_agent("现在几点了？"))
    print(f"\n轨迹已写入：{TRACE_FILE}")
    print("\n".join(TRACE_FILE.read_text(encoding="utf-8").splitlines()[-4:]))


if __name__ == "__main__":
    main()
