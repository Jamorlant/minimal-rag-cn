"""RAG 主流程编排：检索 → 混合重排 → 生成 → 引用校验。

`retrieve()` 是唯一检索入口，基线与完整流程共用它，只改变 k 与是否强制引用，
保证评测对比的差异只来自这两项。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .citations import VerificationResult, verify_answer
from .config import Config
from .embeddings import NgramTfidfEmbedder
from .generator import GenerationResult, make_generator
from .index import load_chunks, load_index
from .logging_utils import get_logger
from .store import RetrievedChunk

logger = get_logger(__name__)


@dataclass
class RagAnswer:
    question: str
    hits: list[RetrievedChunk]
    generation: GenerationResult
    verification: VerificationResult
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "notes": self.notes,
            "retrieved": [
                {
                    "rank": hit.rank,
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 4),
                    "doc": hit.doc,
                    "heading": hit.heading,
                    "para_ids": hit.para_ids,
                    "text": hit.text,
                }
                for hit in self.hits
            ],
            "generation": self.generation.to_dict(),
            "verification": self.verification.to_dict(),
        }


class RagPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder, self.store, self.meta = load_index(cfg)
        self.notes: list[str] = []
        for component in ("embedder", "store"):
            info = self.meta.get(component, {})
            if info.get("degraded"):
                self.notes.append(info.get("note", ""))
        self._lex_embedder: NgramTfidfEmbedder | None = None
        self._lex_vectors: np.ndarray | None = None
        self._lex_rows: dict[str, int] = {}
        if not isinstance(self.embedder, NgramTfidfEmbedder):
            self._init_lexical_reranker()

    def _init_lexical_reranker(self) -> None:
        """稠密召回后叠加字符 n-gram 字面分：术语型问题靠字面命中补足语义模型的模糊性。

        字面分与离线兜底共用同一套哈希实现，索引加载时在内存算一次，查询期只做点积。
        """
        rows = load_chunks(self.cfg)
        if not rows:
            return
        lexical = NgramTfidfEmbedder(dim=self.cfg.embed_dim)
        texts = [row["text"] for row in rows]
        lexical.fit(texts)
        self._lex_embedder = lexical
        self._lex_vectors = lexical.encode(texts)
        self._lex_rows = {row["id"]: index for index, row in enumerate(rows)}

    def _fusion_score(self, hit: RetrievedChunk, lexical_query: np.ndarray) -> float:
        if self._lex_vectors is None or self._lex_embedder is None:
            return hit.score
        index = self._lex_rows.get(hit.chunk_id)
        lexical_score = float(self._lex_vectors[index] @ lexical_query) if index is not None else 0.0
        weight = self.cfg.rerank_weight
        return weight * hit.score + (1.0 - weight) * lexical_score

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.cfg.top_k
        query_vector = self.embedder.encode([question], is_query=True)[0]
        pool = min(max(k * 4, 20), max(1, self.store.count()))
        hits = self.store.query(query_vector, pool)
        if not hits:
            return []

        if self._lex_embedder is not None:
            lexical_query = self._lex_embedder.encode([question], is_query=True)[0]
            scores = [self._fusion_score(hit, lexical_query) for hit in hits]
        else:
            scores = [hit.score for hit in hits]

        order = sorted(range(len(scores)), key=lambda index: -scores[index])[:k]
        return [
            replace(hits[index], score=scores[index], rank=rank)
            for rank, index in enumerate(order, start=1)
        ]

    def ask(
        self,
        question: str,
        k: int | None = None,
        require_citation: bool = True,
        generator_mode: str | None = None,
    ) -> RagAnswer:
        hits = self.retrieve(question, k)
        generator, generator_note = make_generator(
            generator_mode or self.cfg.llm_mode,
            require_citation=require_citation,
            embedder=self.embedder,
            min_hit_score=self.cfg.min_hit_score,
        )
        generation = generator.generate(question, hits)
        verification = verify_answer(
            generation.answer,
            {hit.chunk_id: hit.text for hit in hits},
            threshold=self.cfg.support_threshold,
        )
        logger.debug(
            "问答完成：召回 %d 块，拒答=%s，支撑率=%.3f",
            len(hits),
            generation.refused,
            verification.support_rate,
        )
        return RagAnswer(
            question=question,
            hits=hits,
            generation=generation,
            verification=verification,
            notes=[note for note in (generator_note, generation.note) if note],
        )
