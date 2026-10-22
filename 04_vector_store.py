"""要求 2：向量检索 —— 输入查询文本与 03 的向量化结果，分别用常见方式匹配两路向量，融合排序返回 Top-K：
  1. 稠密匹配：bge 归一化向量点积（等价于余弦相似度）
  2. 稀疏匹配：BM25 分数（jieba 分词 + 语料级 IDF，与 03 的稀疏向量同一公式）
  融合：BM25 分数 min-max 归一化后与稠密余弦加权求和（默认各 0.5），按融合相似度降序取 Top-K。

输出 [(相似度, 拼接信息), ...]，拼接信息 = 标题链（行 x-y）+ content。

实现逻辑写在函数内，文件末尾以 __main__ 入口直接调用函数演示（python 04_vector_store.py），便于 tests 导入。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_embs = None  # 03 模块懒加载缓存（03 是数字开头文件，无法常规 import）


def _get_embeddings_module():
    """按路径加载 03_embeddings 模块（复用其模型加载、分词与 BM25 索引构建）。"""
    global _embs
    if _embs is None:
        # 优先复用 07 等调用方已加载并注册到 sys.modules 的 03 实例，避免 bge 模型重复加载
        cached = sys.modules.get("03_embeddings")
        if cached is not None:
            _embs = cached
            return _embs
        spec = importlib.util.spec_from_file_location(
            "embeddings_03", str(Path(__file__).parent / "03_embeddings.py")
        )
        _embs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_embs)
    return _embs


def _format_item(item: dict) -> str:
    """把 03 输出项的 content/parent_headings/position 拼接为一段可读文本。"""
    chain = " > ".join(item["parent_headings"])
    start, end = item["position"]
    head = f"{chain}（行 {start}-{end}）\n" if chain else f"（行 {start}-{end}）\n"
    return head + item["content"]


def search(
    query: str, items: list[dict], top_k: int = 5, dense_weight: float = 0.5
) -> list[tuple[float, str]]:
    """向量检索（本文件对外主入口）。

    输入：query 检索文本；items 为 03 embed_chunks 的输出列表。
    输出：[(融合相似度, 拼接信息), ...]，按相似度降序，最多 top_k 条。
    dense_weight 为稠密分数占比（稀疏占 1 - dense_weight）。
    """
    if not items:
        return []
    embs = _get_embeddings_module()

    # 稠密路：query 与所有块均为归一化向量，点积即余弦相似度
    dense_q = embs._get_dense_model().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0]
    dense_scores = np.array([float(np.dot(dense_q, it["dense"])) for it in items])

    # 稀疏路：按语料重建 BM25 索引（IDF 语料级统计），对查询分词打分
    bm25 = embs.build_sparse_index([it["content"] for it in items])
    sparse_scores = bm25.get_scores(embs._tokenize(query))

    # BM25 分数无上界，min-max 归一化到 [0,1] 后与稠密余弦加权融合
    lo, hi = float(sparse_scores.min()), float(sparse_scores.max())
    sparse_norm = (sparse_scores - lo) / (hi - lo) if hi > lo else np.zeros_like(sparse_scores)
    fused = dense_weight * dense_scores + (1 - dense_weight) * sparse_norm

    order = np.argsort(-fused)[: max(1, min(top_k, len(items)))]
    return [(float(fused[i]), _format_item(items[i])) for i in order]


def main() -> None:
    """以函数调用方式演示：01 导入 → 02 切分 → 03 双向量化 → 04 检索的完整链路。"""
    # 01/02/03 是数字开头的文件，无法常规 import，用 importlib 按路径加载
    base = Path(__file__).parent
    mods = {}
    for name in ("01_document_loader", "02_chunking", "03_embeddings"):
        spec = importlib.util.spec_from_file_location(name, str(base / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod

    # 演示用 PDF：项目根目录的「AI应用开发工程师——比特鹰成长计划.pdf」
    demo_path = base / "AI应用开发工程师——比特鹰成长计划.pdf"
    md_text = mods["01_document_loader"].load_document(demo_path)
    items = mods["03_embeddings"].embed_chunks(mods["02_chunking"].chunk_markdown(md_text))

    for query in ("什么是上下文窗口", "RAG 的完整流程是什么"):
        print(f"\n=== 查询：{query} ===")
        for score, info in search(query, items, top_k=3):
            first_line = info.splitlines()[0]
            print(f"  {score:.4f}  {first_line}")


if __name__ == "__main__":
    main()
