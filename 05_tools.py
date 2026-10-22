"""要求 3：本地工具集 —— 1) 当前时间查询；2) 文件查询（data/docs 下所有 PDF，返回绝对路径）。

每个工具提供三件套：Python 函数本体、OpenAI tool calling 的 schema（TOOL_SCHEMAS）、
统一执行入口 dispatch_tool（按函数名执行并把结果序列化为 JSON 字符串，供 07 Agent 的
工具调用循环直接回传给模型），函数也可独立调用。

实现逻辑写在函数内，文件末尾以 __main__ 入口直接调用函数演示（python 05_tools.py），便于 tests 导入。
"""

import json
from datetime import datetime
from pathlib import Path

# 知识库目录（相对本文件，从任何工作目录运行都能找到）
DOCS_DIR = Path(__file__).parent / "data" / "docs"


def query_current_time() -> str:
    """当前时间查询：返回本地日期时间字符串（含星期几）。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}（{weekdays[now.weekday()]}）"


def query_pdf_files() -> list[str]:
    """文件查询：返回 data/docs 目录下所有 PDF 文件的绝对路径列表（不递归、按名称排序）。"""
    return sorted(str(p.resolve()) for p in DOCS_DIR.glob("*.pdf"))


# OpenAI tool calling 的工具 schema（与上面的函数一一对应，供 07 Agent 传给模型）
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_current_time",
            "description": "查询当前的本地日期和时间。适用于任何关于现在几点、今天日期、星期几的问题。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_pdf_files",
            "description": "列出知识库目录 data/docs 中所有 PDF 文件的绝对路径。适用于查找知识库里有哪些文档的问题。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 工具名 -> 函数映射（dispatch_tool 用）
_TOOL_FUNCS = {
    "query_current_time": query_current_time,
    "query_pdf_files": query_pdf_files,
}


def dispatch_tool(name: str, arguments: str | dict = "{}") -> str:
    """按工具名执行本地工具，返回 JSON 字符串结果（Agent 工具调用循环的统一入口）。

    未知工具 / 参数错误 / 执行异常均不抛出，返回 {"error": ...} 字符串由模型自行判断，
    保证 Agent 主循环不会因单次工具失败而中断。
    """
    if name not in _TOOL_FUNCS:
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        result = _TOOL_FUNCS[name](**args)
    except Exception as e:  # 工具执行失败也以结果形式回传，交由模型处理
        return json.dumps({"error": f"工具执行失败：{e}"}, ensure_ascii=False)
    return json.dumps({"result": result}, ensure_ascii=False)


def main() -> None:
    """以函数调用方式演示：直接调用两个工具 + 经 dispatch_tool 模拟 Agent 的调用方式。"""
    print("=== 1. 直接调用 ===")
    print(f"当前时间：{query_current_time()}")
    for p in query_pdf_files():
        print(f"PDF 文件：{p}")

    print("\n=== 2. dispatch_tool 调用（模拟 Agent 回传给模型的格式）===")
    print(f"query_current_time -> {dispatch_tool('query_current_time')}")
    print(f"query_pdf_files    -> {dispatch_tool('query_pdf_files')}")
    print(f"未知工具           -> {dispatch_tool('no_such_tool')}")


if __name__ == "__main__":
    main()
