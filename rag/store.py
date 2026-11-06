"""向量存储：默认 ChromaDB 本地持久化，不可用时降级为 numpy 精确检索。

两种实现都只暴露 add / query / count 三个方法，上层（pipeline）不感知差异。
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import Chunk
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc: str
    heading: str
    para_ids: list[str]
    score: float
    rank: int = 0


def _retrieved(
    chunk_id: str,
    text: str,
    doc: str,
    heading: str,
    para_ids: list[str],
    score: float,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        doc=doc,
        heading=heading,
        para_ids=para_ids,
        score=score,
        rank=rank,
    )


class BaseStore:
    kind = "base"
    degraded = False
    note = ""

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaStore(BaseStore):
    kind = "chromadb"

    def __init__(self, path: Path, collection: str = "chunks", reset: bool = False):
        import chromadb
        from chromadb.config import Settings

        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception as exc:  # 首次构建时集合本就不存在，属预期分支
                logger.debug("忽略 delete_collection 失败：%s", exc)
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=[vector.tolist() for vector in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc": chunk.doc,
                    "heading": chunk.heading,
                    "para_ids": json.dumps(chunk.para_ids, ensure_ascii=False),
                }
                for chunk in chunks
            ],
        )

    def query(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        n_results = min(k, self.count())
        if n_results <= 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(result["ids"][0]):
            metadata: dict[str, Any] = result["metadatas"][0][index]
            hits.append(
                _retrieved(
                    chunk_id=chunk_id,
                    text=result["documents"][0][index],
                    doc=metadata.get("doc", ""),
                    heading=metadata.get("heading", ""),
                    para_ids=json.loads(metadata.get("para_ids", "[]")),
                    score=float(1.0 - result["distances"][0][index]),  # cosine 距离 → 相似度
                    rank=index + 1,
                )
            )
        return hits

    def count(self) -> int:
        return self.collection.count()


class NumpyStore(BaseStore):
    """零依赖兜底：全量矩阵点积 + argsort，适合万级以内的块规模。"""

    kind = "numpy"
    degraded = True

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.matrix_path = path / "embeddings.npy"
        self.rows_path = path / "rows.jsonl"
        self._vectors: np.ndarray | None = None
        self._rows: list[dict[str, Any]] = []
        if self.matrix_path.exists() and self.rows_path.exists():
            self.load()

    def load(self) -> None:
        self._vectors = np.load(self.matrix_path).astype(np.float32)
        self._rows = [
            json.loads(line)
            for line in self.rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._vectors = np.asarray(embeddings, dtype=np.float32)
        self._rows = [
            {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "doc": chunk.doc,
                "heading": chunk.heading,
                "para_ids": chunk.para_ids,
            }
            for chunk in chunks
        ]
        np.save(self.matrix_path, self._vectors)
        with self.rows_path.open("w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def query(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        if self._vectors is None:
            self.load()
        if self._vectors is None or not self._rows:
            return []
        scores = self._vectors @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-scores)[: min(k, len(scores))]
        hits: list[RetrievedChunk] = []
        for rank, index in enumerate(top, start=1):
            row = self._rows[int(index)]
            hits.append(
                _retrieved(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    doc=row["doc"],
                    heading=row["heading"],
                    para_ids=row["para_ids"],
                    score=float(scores[int(index)]),
                    rank=rank,
                )
            )
        return hits

    def count(self) -> int:
        return len(self._rows)


class StoreFactory:
    @staticmethod
    def create(kind: str, path: Path, reset: bool = False) -> tuple[BaseStore, str]:
        """返回 (store, 降级说明)；降级说明为空表示按首选实现运行。"""
        if kind == "numpy":
            return NumpyStore(path), ""
        try:
            return ChromaStore(path, reset=reset), ""
        except Exception as exc:  # 导入失败 / 原生库加载失败 / 目录权限
            if kind == "chroma":
                raise
            note = f"降级：ChromaDB 不可用（{type(exc).__name__}: {exc}），改用 numpy 精确索引"
            logger.warning(note)
            return NumpyStore(path), note
