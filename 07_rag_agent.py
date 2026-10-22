"""要求 4+5（综合）：RAG Agent 主循环 —— 综合使用 01-06 的全部能力：
  知识库构建：01 导入 PDF 转 Markdown → 02 最小章节切分 → 03 稠密+稀疏双向量化（进程内缓存一次构建）；
  工具集：05 的 query_current_time / query_pdf_files，加上把 04 的混合检索包装成的 knowledge_search 工具；
  主循环：模型自主决策「检索知识库 / 调用工具 / 直接回答」，经 06 的 @trace_agent 装饰器把每步
  reasoning 摘要、工具调用、最终结果写入 data/traces.jsonl；回答要求标注引用来源，检索无果时拒答。

实现逻辑写在函数内，文件末尾以 __main__ 入口运行演示（python 07_rag_agent.py，走真实 API），便于 tests 导入。
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 01-06 是数字开头的文件，无法常规 import，用 importlib 按路径加载（进程内只加载一次）
base = Path(__file__).parent
_siblings: dict = {}


def _sib(name: str):
    """按名字取已加载的兄弟模块（01-06），首次访问时加载。"""
    if name not in _siblings:
        spec = importlib.util.spec_from_file_location(name, str(base / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # 注册到 sys.modules，供 04 等兄弟模块复用同一实例，避免模型重复加载
        spec.loader.exec_module(mod)
        _siblings[name] = mod
    return _siblings[name]


load_dotenv()  # 从项目根 .env 读取配置

SYSTEM_PROMPT = """你是「比特鹰学习助手」，基于本地知识库回答 AI 应用开发学习问题。回答规则：
1. 涉及课程资料、知识点、学习安排 → 调用 knowledge_search 检索知识库，基于检索结果回答，
   并在引用处标注来源，格式如【来源：# 标题（行 x-y）】；
2. 检索结果的 similarity 全部低于 0.40，或与问题明显无关 → 回答"知识库中没有找到相关内容"并拒答，
   不要编造；
3. 涉及当前时间/日期/星期 → 调用 query_current_time；
4. 询问知识库里有哪些文档 → 调用 query_pdf_files；
5. 一般常识、闲聊 → 直接回答，不必调用任何工具。"""

# 单次提问允许的最大「工具调用轮数」，防止模型无限循环调用
MAX_TOOL_ROUNDS = 4

# 知识库缓存（进程内只构建一次：01 导入 → 02 切分 → 03 双向量化）
_items: list[dict] | None = None


def _get_items() -> list[dict]:
    """构建并缓存知识库（04 检索所需的向量化条目列表）。"""
    global _items
    if _items is None:
        loader = _sib("01_document_loader")
        chunker = _sib("02_chunking")
        embeddings = _sib("03_embeddings")
        demo_path = base / "AI应用开发工程师——比特鹰成长计划.pdf"
        chunks = chunker.chunk_markdown(loader.load_document(demo_path))
        _items = embeddings.embed_chunks(chunks)
    return _items


def knowledge_search(query: str, top_k: int = 3) -> str:
    """知识库混合检索（把 04 的 search 包装成 Agent 工具，返回 JSON 字符串）。"""
    results = _sib("04_vector_store").search(query, _get_items(), top_k=top_k)
    payload = [{"similarity": round(score, 4), "text": text} for score, text in results]
    return json.dumps({"result": payload}, ensure_ascii=False)


# Agent 可用工具 = 05 的两个本地工具 + knowledge_search（schema 与执行一一对应）
AGENT_TOOL_SCHEMAS = [*_sib("05_tools").TOOL_SCHEMAS, {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": "在本地知识库（比特鹰课程资料）中检索与查询最相关的章节，返回相似度与原文片段。回答任何课程知识问题前都应先调用它。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索文本，用与问题相关的关键词或短语"},
                "top_k": {"type": "integer", "description": "返回条数，默认 3"},
            },
            "required": ["query"],
        },
    },
}]


def dispatch_agent_tool(name: str, arguments: str | dict) -> str:
    """Agent 的统一工具执行入口：05 的两个工具走 05.dispatch_tool，knowledge_search 本地执行。"""
    if name == "knowledge_search":
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        try:
            return knowledge_search(**args)
        except Exception as e:  # 与 05.dispatch_tool 同策略：失败以结果回传，不中断主循环
            return json.dumps({"error": f"工具执行失败：{e}"}, ensure_ascii=False)
    return _sib("05_tools").dispatch_tool(name, arguments)


def get_client() -> OpenAI:
    """创建 OpenAI 客户端（阿里云 MaaS 兼容接口），模型从 .env 的 LLM_QWEN_MODEL 读取。"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("OpenAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("请先在 .env 中配置 API_KEY 与 OpenAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


@_sib("06_trace_logging").trace_agent
def ask(question: str, client: OpenAI, tracer) -> str:
    """单轮 Agent 主循环（06 装饰器自动记录 start / final_result）。

    每轮把带工具 schema 的请求发给模型：模型发起工具调用则执行并把结果回传继续，
    不再调用工具则视为最终回答。
    """
    tracer.log_reasoning("收到问题，等待模型决策是否检索/调用工具")
    model = os.getenv("LLM_QWEN_MODEL")
    if not model:
        raise RuntimeError("请先在 .env 中配置 LLM_QWEN_MODEL")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=AGENT_TOOL_SCHEMAS
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:  # 模型不再调用工具 → 最终回答
            tracer.log_reasoning("模型已给出最终回答（无更多工具调用）")
            return msg.content

        names = [tc.function.name for tc in msg.tool_calls]
        tracer.log_reasoning(f"模型发起工具调用：{', '.join(names)}")
        messages.append(msg)  # assistant 消息（含 tool_calls）原样回传，保持上下文
        for tc in msg.tool_calls:
            result = dispatch_agent_tool(tc.function.name, tc.function.arguments)
            tracer.log_tool(tc.function.name, tc.function.arguments, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "抱歉，工具调用轮次超限，未能完成回答。请尝试换一种问法。"


def main() -> None:
    """以函数调用方式演示：四类问题（知识库检索 / 时间 / 文档清单 / 超纲拒答）。"""
    client = get_client()
    demos = [
        ("知识库检索", "什么是上下文窗口？"),
        ("时间工具", "现在几点了？"),
        ("文件工具", "知识库里有哪些 PDF 文档？"),
        ("超纲拒答", "今天上证指数收盘多少点？"),
    ]
    for tag, question in demos:
        print(f"\n=== {tag}：{question} ===")
        print(ask(question, client))


if __name__ == "__main__":
    main()
