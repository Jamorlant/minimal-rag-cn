"""向量库：默认 ChromaDB 本地持久化，安装失败时降级为 numpy 精确检索。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc: str
    heading: str
    para_ids: List[str]
    score: float
    rank: int = 0


class BaseStore:
    kind = "base"
    degraded = False
    note = ""

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, k: int) -> List[RetrievedChunk]:
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
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[v.tolist() for v in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc": c.doc, "heading": c.heading, "para_ids": json.dumps(c.para_ids, ensure_ascii=False)}
                for c in chunks
            ],
        )

    def query(self, vector: np.ndarray, k: int) -> List[RetrievedChunk]:
        n = min(k, self.count())
        if n <= 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out: List[RetrievedChunk] = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=res["documents"][0][i],
                    doc=meta.get("doc", ""),
                    heading=meta.get("heading", ""),
                    para_ids=json.loads(meta.get("para_ids", "[]")),
                    score=float(1.0 - res["distances"][0][i]),  # cosine 距离转相似度
                    rank=i + 1,
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()


class NumpyStore(BaseStore):
    """零依赖兜底实现：全量矩阵点积 + argsort，适合万级以内的块规模。"""

    kind = "numpy"
    degraded = True

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.matrix_path = path / "embeddings.npy"
        self.rows_path = path / "rows.jsonl"
        self._vectors: np.ndarray | None = None
        self._rows: List[dict] = []
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
                "chunk_id": c.id,
                "text": c.text,
                "doc": c.doc,
                "heading": c.heading,
                "para_ids": c.para_ids,
            }
            for c in chunks
        ]
        np.save(self.matrix_path, self._vectors)
        with self.rows_path.open("w", encoding="utf-8") as f:
            for row in self._rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def query(self, vector: np.ndarray, k: int) -> List[RetrievedChunk]:
        if self._vectors is None:
            self.load()
        if self._vectors is None or len(self._rows) == 0:
            return []
        scores = self._vectors @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-scores)[: min(k, len(scores))]
        out: List[RetrievedChunk] = []
        for rank, idx in enumerate(top, start=1):
            row = self._rows[int(idx)]
            out.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    doc=row["doc"],
                    heading=row["heading"],
                    para_ids=row["para_ids"],
                    score=float(scores[int(idx)]),
                    rank=rank,
                )
            )
        return out

    def count(self) -> int:
        return len(self._rows)


class StoreFactory:
    @staticmethod
    def create(kind: str, path: Path, reset: bool = False) -> Tuple[BaseStore, str]:
        """返回 (store, 降级说明)。"""
        if kind == "numpy":
            return NumpyStore(path), ""
        try:
            return ChromaStore(path, reset=reset), ""
        except Exception as exc:
            if kind == "chroma":
                raise
            store = NumpyStore(path)
            return store, f"降级：ChromaDB 不可用（{type(exc).__name__}: {exc}），改用 numpy 精确索引"
