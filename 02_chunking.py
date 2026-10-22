"""要求 2：文档切分 —— 按"最小章节"（最深层级叶子标题下的正文）切分 01 输出的 Markdown 全文。

每个切分块为三元组：
  1. content        最小章节的正文内容（不含任何标题行）
  2. parent_headings 父子标题链（从 # 到叶子标题，按层级排列，如 ["# LLM 基础", "### 大模型基本原理"]）
  3. position       内容在当前文档中的行号区间 (start_line, end_line)，1-based 闭区间，
                    该行区间（含首尾行内空白）剥离空白后即为 content

实现逻辑写在函数内，文件末尾以 __main__ 入口直接调用函数演示（python 02_chunking.py），便于 tests 导入。
"""

import importlib.util
import re
from pathlib import Path

# 标题行：1-6 个 # + 至少一个空白（Markdown atx 标准；代码围栏内的行会被跳过）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _heading_level(line: str) -> int | None:
    """返回该行的标题级别（1-6 对应 #-######），非标题行返回 None。"""
    m = _HEADING_RE.match(line)
    return len(m.group(1)) if m else None


def chunk_markdown(md_text: str) -> list[tuple[str, list[str], tuple[int, int]]]:
    """把 Markdown 全文按最小章节切分（02 对外主入口）。

    切分规则：以最深层级的叶子章节为块，父级标题不单独成块（父级无直接正文时不产生块）；
    遇到任意级别标题即结束上一个正文块，代码围栏（```）内的 # 不视为标题。

    返回 [(content, parent_headings, (start_line, end_line)), ...]，三个字段含义见模块 docstring。
    注意：行号基于 \\n 规范化后的文本计算（\r\n 会先被替换为 \\n），1-based 闭区间。
    """
    md_text = md_text.replace("\r\n", "\n")
    chunks: list[tuple[str, list[str], tuple[int, int]]] = []
    headings: list[str] = []  # 当前标题链，每项为标题行原文（如 "# LLM 基础"）
    buf_start_line: int | None = None  # 当前正文缓冲起始行号（1-based）；None 表示缓冲为空
    buf_lines: list[str] = []
    line_no = 0
    in_code_fence = False

    def flush() -> None:
        """把当前正文缓冲落为一个切分块（内容为空则丢弃），行号区间收缩到首尾非空行。"""
        nonlocal buf_start_line, buf_lines
        if buf_start_line is None:
            return
        non_empty = [i for i, l in enumerate(buf_lines) if l.strip()]
        if non_empty:  # 缓冲全是空白行则不产生块
            first, last = non_empty[0], non_empty[-1]
            content = "\n".join(buf_lines[first : last + 1]).strip()
            chunks.append((content, list(headings), (buf_start_line + first, buf_start_line + last)))
        buf_start_line = None
        buf_lines = []

    for line in md_text.splitlines():
        line_no += 1  # 当前行号（1-based）
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
        level = None if in_code_fence else _heading_level(line)
        if level is not None:
            flush()  # 之前缓冲的正文属于上一个叶子标题，结束于上一行
            del headings[level - 1 :]  # 回退到比当前标题更浅的层级
            headings.append(line.strip())
        else:
            if buf_start_line is None:
                buf_start_line = line_no
            buf_lines.append(line)

    flush()  # 文档结尾的最后一块
    return chunks


def main() -> None:
    """以函数调用方式演示：经 01 导入演示 PDF 转为 Markdown 后按最小章节切分。"""
    # 01 是数字开头的文件，无法常规 import，用 importlib 按路径加载
    loader_path = Path(__file__).parent / "01_document_loader.py"
    spec = importlib.util.spec_from_file_location("document_loader", str(loader_path))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)

    # 演示用 PDF：项目根目录的「AI应用开发工程师——比特鹰成长计划.pdf」
    demo_path = Path(__file__).parent / "AI应用开发工程师——比特鹰成长计划.pdf"
    md_text = loader.load_document(demo_path)  # 用 load_document 避免 main 演示触发备份

    chunks = chunk_markdown(md_text)
    print(f"=== 切分结果：共 {len(chunks)} 个最小章节 ===")
    for content, parent_headings, (start_line, end_line) in chunks[:3]:
        chain = " > ".join(parent_headings) or "（文档开头，无标题）"
        preview = content[:80].replace("\n", " ")
        print(f"\n--- 标题链：{chain}")
        print(f"    行号：{start_line}-{end_line}")
        print(f"    内容预览：{preview}...")


if __name__ == "__main__":
    main()
