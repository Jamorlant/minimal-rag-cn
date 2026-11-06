"""向量化：优先 BGE 中文小模型，不可用时降级为离线字符 n-gram 哈希 TF-IDF。

降级只发生一次（构建/加载索引时），原因写入 note 并随 meta 落盘，运行期不再切换。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .logging_utils import get_logger
from .textutil import char_ngrams, normalize, stable_hash

logger = get_logger(__name__)


class BaseEmbedder:
    name: str = "base"
    dim: int = 0
    degraded: bool = False
    note: str = ""

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def fit(self, texts: Sequence[str]) -> None:
        """离线方案用于统计 IDF，稠密模型无需实现。"""


class SentenceTransformerEmbedder(BaseEmbedder):
    """BAAI/bge-small-zh-v1.5，查询侧加官方推荐指令前缀。"""

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        # 先只读本地缓存：离线环境和重复加载都不会去探测 huggingface.co
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception as exc:
            logger.warning("本地缓存加载失败，尝试联网下载 %s：%s", model_name, exc)
            self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.note = "sentence-transformers 稠密向量"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prepared = [self.QUERY_PREFIX + text if is_query else text for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class NgramTfidfEmbedder(BaseEmbedder):
    """离线兜底：字符 2/3-gram + 带符号哈希 + IDF 加权 + L2 归一化。

    不依赖模型文件与网络，确定性可复现；语义能力弱于稠密模型，但中文短查询召回稳定。
    """

    def __init__(self, dim: int = 2048, min_df: int = 2):
        self.name = "offline-char-ngram-tfidf"
        self.dim = dim
        self.min_df = min_df
        self.degraded = True
        self.note = "降级方案：字符 n-gram 哈希 TF-IDF，无预训练语义能力"
        self.idf = np.ones(dim, dtype=np.float32)
        self._fitted = False

    def _tokens(self, text: str) -> list[str]:
        compact = normalize(text)
        return char_ngrams(compact, 2) + char_ngrams(compact, 3)

    def fit(self, texts: Sequence[str]) -> None:
        document_frequency = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            for bucket in {stable_hash(token, self.dim) for token in self._tokens(text)}:
                document_frequency[bucket] += 1
        # 标准平滑 IDF；未出现过的桶保留基础权重
        self.idf = np.log((len(texts) + 1) / (document_frequency + 1)).astype(np.float32) + 1.0
        self._fitted = True

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("离线向量器尚未 fit，请先构建索引")
        output = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            grams = self._tokens(text)
            if not grams:
                continue
            counts: dict[int, float] = {}
            for token in grams:
                index = stable_hash(token, self.dim)
                sign = 1.0 if stable_hash(token + "#s", 2) == 0 else -1.0
                counts[index] = counts.get(index, 0.0) + sign
            for index, term_frequency in counts.items():
                # 次线性 TF，抑制长文本重复词的权重膨胀；符号由稳定哈希决定，用于抵消共现偏差
                output[row, index] = (
                    (1.0 + np.log1p(abs(term_frequency)))
                    * np.sign(term_frequency)
                    * self.idf[index]
                )
            norm = np.linalg.norm(output[row])
            if norm > 0:
                output[row] /= norm
        return output

    def save_idf(self, path: Path) -> None:
        np.save(path, self.idf)

    def load_idf(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"IDF 文件不存在：{path}，请重新构建索引")
        self.idf = np.load(path).astype(np.float32)
        self._fitted = True


class EmbedderFactory:
    """按配置构造向量器；稠密模型不可用时自动降级并记录原因。"""

    @staticmethod
    def create(
        mode: str = "auto",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 2048,
    ) -> BaseEmbedder:
        if mode == "ngram":
            return NgramTfidfEmbedder(dim=dim)
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception as exc:  # 包缺失 / 权重下载失败 / 离线无缓存
            if mode == "st":
                raise
            logger.warning("稠密向量不可用，降级为 n-gram TF-IDF：%s: %s", type(exc).__name__, exc)
            embedder = NgramTfidfEmbedder(dim=dim)
            embedder.note = (
                f"降级方案：字符 n-gram 哈希 TF-IDF（BGE 不可用：{type(exc).__name__}: {exc}）"
            )
            return embedder


def describe(embedder: BaseEmbedder) -> dict[str, object]:
    return {
        "kind": type(embedder).__name__,
        "name": embedder.name,
        "dim": embedder.dim,
        "degraded": embedder.degraded,
        "note": embedder.note,
    }
