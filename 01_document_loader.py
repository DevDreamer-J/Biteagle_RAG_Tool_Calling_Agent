"""要求 1：文档导入 —— 输入 Markdown / PDF 文档的本地绝对路径，统一输出 Markdown 全文文本字符串：
PDF 用 pymupdf4llm 转换为 Markdown，Markdown 文件直接读入；
同时把原文件按原文档格式备份到 data/docs/（保留原文件名），重名时返回错误信息、不覆盖。

实现逻辑写在函数内，文件末尾以 __main__ 入口直接调用函数演示（python 01_document_loader.py），便于 tests 导入。
"""

import shutil
from pathlib import Path

import pymupdf4llm

# 备份目录（相对于本文件，从任何工作目录运行都能找到）
DOCS_DIR = Path(__file__).parent / "data" / "docs"

# 支持导入的文档后缀
SUPPORTED_SUFFIXES = {".md", ".pdf"}


def _validate_document_path(path: str | Path) -> Path:
    """校验输入：必须是本地绝对路径、真实存在、后缀为 Markdown/PDF，返回 Path 对象，不合法抛异常。"""
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(f"请传入本地绝对路径，收到：{path}")
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在：{p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = " / ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"仅支持 {supported} 文档，收到：{p.name}")
    return p


def load_pdf_as_markdown(path: str | Path) -> str:
    """加载 PDF 文件，用 pymupdf4llm 转换为 Markdown（含标题层级、列表、表格），返回 Markdown 字符串。"""
    p = _validate_document_path(path)
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持 PDF 文件，收到：{p.name}")
    return pymupdf4llm.to_markdown(str(p)).strip()


def load_markdown(path: str | Path) -> str:
    """加载 Markdown 文件，按 UTF-8 读入，返回全文文本字符串。"""
    p = _validate_document_path(path)
    if p.suffix.lower() != ".md":
        raise ValueError(f"仅支持 Markdown 文件，收到：{p.name}")
    return p.read_text(encoding="utf-8").strip()


def load_document(path: str | Path) -> str:
    """加载 Markdown / PDF 文档，按后缀分流，统一返回 Markdown 全文文本字符串。"""
    p = _validate_document_path(path)
    if p.suffix.lower() == ".pdf":
        return load_pdf_as_markdown(p)
    return load_markdown(p)


def backup_document(path: str | Path) -> tuple[bool, str]:
    """把原文件备份到 data/docs/，文件名保持原名。

    返回 (是否成功, 信息)：
      成功 -> (True, "已备份到 <目标路径>")
      重名 -> (False, "错误：备份目录已存在同名文件「<name>」，未覆盖")
    """
    p = _validate_document_path(path)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCS_DIR / p.name
    if target.exists():
        return False, f"错误：备份目录已存在同名文件「{p.name}」，未覆盖（{target}）"
    shutil.copy2(p, target)  # copy2 保留文件元数据
    return True, f"已备份到 {target}"


def import_document(path: str | Path) -> str:
    """导入 Markdown / PDF 文档（本文件对外主入口）。

    成功：按原文档格式完成备份，并返回 Markdown 全文文本字符串；
    重名：不覆盖已有备份，返回错误信息字符串。
    """
    p = _validate_document_path(path)
    text = load_document(p)  # 先提取，解析失败直接抛异常、不产生备份
    ok, msg = backup_document(p)  # 再备份；重名时返回错误信息
    if not ok:
        return msg
    return text


def main() -> None:
    """以函数调用方式演示本文件功能：把 demo_path 换成任意本地 Markdown/PDF 绝对路径即可。

    首次运行：提取全文并备份到 data/docs/；再次运行：返回重名错误信息。
    """
    # 演示用 PDF：项目根目录的「AI应用开发工程师——比特鹰成长计划.pdf」
    demo_path = Path(__file__).parent / "AI应用开发工程师——比特鹰成长计划.pdf"

    print("=== 导入结果（全文文本或错误信息）===")
    print(import_document(demo_path))


if __name__ == "__main__":
    main()
