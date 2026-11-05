"""RAG 主流程：检索 -> 生成 -> 引用校验。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .citations import VerificationResult, verify_answer
from .config import Config
from .embeddings import NgramTfidfEmbedder
from .generator import GenerationResult, make_generator
from .index import load_chunks, load_index
from .store import RetrievedChunk


@dataclass
class RagAnswer:
    question: str
    hits: List[RetrievedChunk]
    generation: GenerationResult
    verification: VerificationResult
    notes: List[str]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "notes": self.notes,
            "retrieved": [
                {
                    "rank": h.rank,
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 4),
                    "doc": h.doc,
                    "heading": h.heading,
                    "para_ids": h.para_ids,
                    "text": h.text,
                }
                for h in self.hits
            ],
            "generation": self.generation.to_dict(),
            "verification": self.verification.to_dict(),
        }


class RagPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder, self.store, self.meta = load_index(cfg)
        self.notes: List[str] = []
        if self.meta.get("embedder", {}).get("degraded"):
            self.notes.append(self.meta["embedder"].get("note", ""))
        if self.meta.get("store", {}).get("degraded"):
            self.notes.append(self.meta["store"].get("note", ""))
        self._lex_embedder = None
        self._lex_vectors = None
        self._lex_rows: dict = {}
        if not isinstance(self.embedder, NgramTfidfEmbedder):
            self._init_lexical_reranker()

    def _init_lexical_reranker(self) -> None:
        """稠密召回后叠加字符 n-gram 字面分：术语型问题靠字面命中补足语义模型的模糊性。"""
        rows = load_chunks(self.cfg)
        if not rows:
            return
        lex = NgramTfidfEmbedder(dim=self.cfg.embed_dim)
        texts = [r["text"] for r in rows]
        lex.fit(texts)
        self._lex_embedder = lex
        self._lex_vectors = lex.encode(texts)
        self._lex_rows = {r["id"]: i for i, r in enumerate(rows)}

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        k = k or self.cfg.top_k
        vector = self.embedder.encode([question], is_query=True)[0]
        pool = min(max(k * 4, 20), max(1, self.store.count()))
        hits = self.store.query(vector, pool)
        if self._lex_embedder is None or not hits:
            return hits[:k]
        q_lex = self._lex_embedder.encode([question], is_query=True)[0]
        weight = self.cfg.rerank_weight
        rescored = []
        for hit in hits:
            idx = self._lex_rows.get(hit.chunk_id)
            lex_score = float(self._lex_vectors[idx] @ q_lex) if idx is not None else 0.0
            rescored.append((weight * hit.score + (1.0 - weight) * lex_score, hit))
        rescored.sort(key=lambda x: -x[0])
        out: List[RetrievedChunk] = []
        for rank, (score, hit) in enumerate(rescored[:k], start=1):
            hit.score = score
            hit.rank = rank
            out.append(hit)
        return out

    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        require_citation: bool = True,
        generator_mode: Optional[str] = None,
    ) -> RagAnswer:
        hits = self.retrieve(question, k)
        generator, gen_note = make_generator(
            generator_mode or self.cfg.llm_mode,
            require_citation=require_citation,
            embedder=self.embedder,
            min_hit_score=self.cfg.min_hit_score,
        )
        generation = generator.generate(question, hits)
        verification = verify_answer(
            generation.answer,
            {h.chunk_id: h.text for h in hits},
            threshold=self.cfg.support_threshold,
        )
        notes = [n for n in [gen_note, generation.note] if n]
        return RagAnswer(
            question=question,
            hits=hits,
            generation=generation,
            verification=verification,
            notes=notes,
        )
