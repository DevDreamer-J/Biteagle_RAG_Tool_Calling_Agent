"""要求 2：向量化 —— 输入 02 切分块 [(content, parent_headings, position), ...]，
为每个块产出两种向量并组装为字典列表：
  1. dense  稠密向量：本地 bge-small-zh-v1.5（sentence-transformers，离线），512 维，已归一化
  2. sparse 稀疏向量：BM25 词权重 {token: weight}（rank-bm25 + jieba 分词，语料级 IDF 现场统计）

输出 [{"sparse", "dense", "content", "parent_headings", "position"}, ...]，保留 02 的全部三条信息。

实现逻辑写在函数内，文件末尾以 __main__ 入口直接调用函数演示（python 03_embeddings.py），便于 tests 导入。
"""

import importlib.util
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 本地 bge-small-zh-v1.5 模型目录（相对本文件，离线加载，不联网、不读环境变量）
EMBEDDING_MODEL_DIR = Path(__file__).parent / "embedding" / "bge-small-zh-v1.5"

_dense_model: SentenceTransformer | None = None  # 懒加载缓存，避免重复加载模型


def _get_dense_model() -> SentenceTransformer:
    """懒加载本地 bge-small-zh-v1.5 模型（首次调用时加载一次）。"""
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(str(EMBEDDING_MODEL_DIR))
    return _dense_model


def _tokenize(text: str) -> list[str]:
    """中文分词（jieba 精确模式），作为 BM25 的词元。"""
    return jieba.lcut(text)


def build_sparse_index(contents: list[str]) -> BM25Okapi:
    """对全部块语料构建 BM25 索引。

    BM25 的 IDF 是语料级统计量，必须对全部文档一次性构建，不能逐块独立计算。
    """
    return BM25Okapi([_tokenize(c) for c in contents])


def sparse_vectorize(bm25: BM25Okapi, doc_idx: int) -> dict[str, float]:
    """取第 doc_idx 篇文档的 BM25 稀疏向量：{token: idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))}。

    与 BM25Okapi.get_score 公式一致——查询词 t 落在本文档时，
    get_scores([t])[doc_idx] 恰等于 sparse[t]（保证检索时可复用同一套权重）。
    """
    dl = bm25.doc_len[doc_idx]
    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    return {
        term: float(bm25.idf[term] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)))
        for term, tf in bm25.doc_freqs[doc_idx].items()
    }


def embed_chunks(chunks: list[tuple[str, list[str], tuple[int, int]]]) -> list[dict]:
    """向量化切分块（本文件对外主入口）。

    输入：02 的 chunk_markdown 输出 [(content, parent_headings, position), ...]；
    输出：[{"sparse": {token: w}, "dense": ndarray(512), "content", "parent_headings", "position"}, ...]，
    顺序与输入一致。
    """
    if not chunks:
        return []
    contents = [c[0] for c in chunks]

    # 稀疏：语料级建索引 + 逐块取权重；稠密：批量编码（bge 官方建议归一化，便于余弦=点积）
    bm25 = build_sparse_index(contents)
    dense = _get_dense_model().encode(contents, normalize_embeddings=True, show_progress_bar=False)

    return [
        {
            "sparse": sparse_vectorize(bm25, i),
            "dense": dense[i].copy(),
            "content": content,
            "parent_headings": parent_headings,
            "position": position,
        }
        for i, (content, parent_headings, position) in enumerate(chunks)
    ]


def main() -> None:
    """以函数调用方式演示：01 导入 → 02 切分 → 03 双向量化的完整链路。"""
    # 01/02 是数字开头的文件，无法常规 import，用 importlib 按路径加载
    base = Path(__file__).parent
    loaded = {}
    for name in ("01_document_loader", "02_chunking"):
        spec = importlib.util.spec_from_file_location(name, str(base / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        loaded[name] = mod

    # 演示用 PDF：项目根目录的「AI应用开发工程师——比特鹰成长计划.pdf」
    demo_path = base / "AI应用开发工程师——比特鹰成长计划.pdf"
    chunks = loaded["02_chunking"].chunk_markdown(loaded["01_document_loader"].load_document(demo_path))

    items = embed_chunks(chunks)
    print(f"=== 向量化结果：共 {len(items)} 个块 ===")
    first = items[0]
    print(f"字典键：{sorted(first.keys())}")
    print(f"稠密向量：shape={first['dense'].shape}，范数={np.linalg.norm(first['dense']):.4f}")
    top_sparse = sorted(first["sparse"].items(), key=lambda kv: -kv[1])[:5]
    print(f"稀疏向量：{len(first['sparse'])} 个词元，权重 top5：{[(t, round(w, 3)) for t, w in top_sparse]}")
    print(f"标题链：{' > '.join(first['parent_headings']) or '（无）'}")
    print(f"位置：行 {first['position'][0]}-{first['position'][1]}")


if __name__ == "__main__":
    main()
