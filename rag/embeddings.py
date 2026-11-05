"""向量化：优先 BGE 中文小模型，不可用时降级为离线字符 n-gram 哈希 TF-IDF。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .textutil import char_ngrams, normalize, stable_hash


class BaseEmbedder:
    name: str = "base"
    dim: int = 0
    degraded: bool = False
    note: str = ""

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def fit(self, texts: Sequence[str]) -> None:
        """离线方案用于统计 IDF，稠密模型无需实现。"""

    def save_meta(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"name": self.name, "dim": self.dim, "degraded": self.degraded, "note": self.note},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class SentenceTransformerEmbedder(BaseEmbedder):
    """BAAI/bge-small-zh-v1.5，查询侧加官方推荐指令前缀。"""

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        # 优先只读本地缓存：离线环境与重复加载都不会去探测 huggingface.co
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.note = "sentence-transformers 稠密向量"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prepared = [
            (self.QUERY_PREFIX + t if is_query else t) for t in texts
        ]
        vectors = self.model.encode(
            prepared, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)


class NgramTfidfEmbedder(BaseEmbedder):
    """离线兜底：字符 2/3-gram + 带符号哈希 + IDF 加权 + L2 归一化。

    不依赖任何模型文件与网络，确定性可复现；语义能力弱于稠密模型，但中文短查询召回稳定。
    """

    def __init__(self, dim: int = 2048, min_df: int = 2):
        self.name = "offline-char-ngram-tfidf"
        self.dim = dim
        self.min_df = min_df
        self.degraded = True
        self.note = "降级方案：字符 n-gram 哈希 TF-IDF，无预训练语义能力"
        self.idf = np.ones(dim, dtype=np.float32)
        self._fitted = False

    def _tokens(self, text: str) -> List[str]:
        compact = normalize(text)
        grams = char_ngrams(compact, 2) + char_ngrams(compact, 3)
        return grams

    def fit(self, texts: Sequence[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            seen = {stable_hash(tok, self.dim) for tok in self._tokens(text)}
            for idx in seen:
                df[idx] += 1
        n = len(texts)
        # 标准平滑 IDF，未出现过的桶给最大权重以外的最小值
        self.idf = np.log((n + 1) / (df + 1)).astype(np.float32) + 1.0
        self._fitted = True

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("离线向量器尚未 fit，请先构建索引")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            grams = self._tokens(text)
            if not grams:
                continue
            counts: dict[int, float] = {}
            for tok in grams:
                idx = stable_hash(tok, self.dim)
                sign = 1.0 if stable_hash(tok + "#s", 2) == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, tf in counts.items():
                # 次线性 TF，抑制长文本重复词的权重膨胀
                out[row, idx] = (1.0 + np.log1p(abs(tf))) * np.sign(tf) * self.idf[idx]
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out

    def save_idf(self, path: Path) -> None:
        np.save(path, self.idf)

    def load_idf(self, path: Path) -> None:
        self.idf = np.load(path).astype(np.float32)
        self._fitted = True


class EmbedderFactory:
    """按配置构造向量器；稠密模型不可用时自动降级并记录原因。"""

    def __new__(
        cls,
        mode: str = "auto",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 2048,
    ) -> BaseEmbedder:
        if mode == "ngram":
            return NgramTfidfEmbedder(dim=dim)
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception as exc:  # 模型包缺失、权重下载失败、离线环境
            if mode == "st":
                raise
            embedder = NgramTfidfEmbedder(dim=dim)
            embedder.note = f"降级方案：字符 n-gram 哈希 TF-IDF（BGE 不可用：{type(exc).__name__}: {exc}）"
            return embedder
