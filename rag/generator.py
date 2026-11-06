"""答案生成：可选 LLM（OpenAI 兼容接口），不可用时抽取式兜底。

两种模式都强制（或显式关闭）引用：
- LlmGenerator：提示词要求每个结论后标注 [chunk_id]，资料不足时拒答；
- ExtractiveGenerator：返回命中最高的原文句子并附块编号与摘录，零幻觉。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .citations import extract_citations
from .logging_utils import get_logger
from .store import RetrievedChunk
from .textutil import overlap_score, split_sentences

logger = get_logger(__name__)
REFUSAL_ANSWER = "资料不足，无法回答该问题。"


@dataclass
class GenerationResult:
    answer: str
    mode: str
    note: str = ""
    citations: list[str] = field(default_factory=list)
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "note": self.note,
            "citations": self.citations,
            "refused": self.refused,
        }


@dataclass
class ScoredSentence:
    score: float
    chunk_id: str
    sentence: str


def _format_context(hits: Sequence[RetrievedChunk]) -> str:
    blocks = [
        f"[{hit.chunk_id}]（来源：{hit.doc} · {hit.heading}）\n{hit.text}" for hit in hits
    ]
    return "\n\n".join(blocks)


class BaseGenerator:
    mode = "base"

    def generate(self, question: str, hits: Sequence[RetrievedChunk]) -> GenerationResult:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """抽取式兜底：不调用任何模型接口，答案即原文句子。"""

    mode = "extractive"

    def __init__(
        self,
        max_sentences: int = 1,
        min_score: float = 0.1,
        require_citation: bool = True,
        embedder: Any = None,
        min_hit_score: float = 0.0,
        keep_ratio: float = 0.92,
        max_chunks: int = 1,
    ):
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.require_citation = require_citation
        self.embedder = embedder
        self.min_hit_score = min_hit_score
        self.keep_ratio = keep_ratio
        # 只从最相关的块内选句：跨块拼接容易把相邻话题的句子凑成答案
        self.max_chunks = max(1, max_chunks)

    def _score_sentences(
        self, question: str, hits: Sequence[RetrievedChunk]
    ) -> list[ScoredSentence]:
        """向量余弦为主、字面覆盖为辅：语义相近但没答到问题焦点的句子会被压低。"""
        candidates: list[tuple[RetrievedChunk, str]] = []
        for hit in list(hits)[: self.max_chunks]:
            for sentence in split_sentences(hit.text):
                candidates.append((hit, sentence))
        if not candidates:
            return []

        if self.embedder is not None:
            vectors = self.embedder.encode([sentence for _, sentence in candidates])
            query_vector = self.embedder.encode([question], is_query=True)[0]
            cosine = vectors @ query_vector
            scored = [
                ScoredSentence(
                    score=float(cos)
                    + 0.2 * overlap_score(sentence, question)
                    + 0.02 / max(1, hit.rank),
                    chunk_id=hit.chunk_id,
                    sentence=sentence,
                )
                for (hit, sentence), cos in zip(candidates, cosine, strict=True)
            ]
            return scored

        return [
            ScoredSentence(
                score=overlap_score(sentence, question) + 0.15 * max(0.0, hit.score),
                chunk_id=hit.chunk_id,
                sentence=sentence,
            )
            for hit, sentence in candidates
        ]

    def _pick(self, candidates: list[ScoredSentence]) -> list[ScoredSentence]:
        picked: list[ScoredSentence] = []
        seen: set[str] = set()
        best = candidates[0].score if candidates else 0.0
        for candidate in candidates:
            if candidate.score < self.min_score:
                continue
            if len(picked) >= self.max_sentences:
                break
            if picked and candidate.score < best * self.keep_ratio:
                break  # 次优句明显不如首句时不再凑数
            key = "".join(candidate.sentence.split())
            if key in seen:  # 重叠切块会产生重复句
                continue
            seen.add(key)
            picked.append(candidate)
        return picked

    def _refuse(self, reason: str) -> GenerationResult:
        return GenerationResult(
            answer=REFUSAL_ANSWER,
            mode=self.mode,
            note=reason,
            citations=[],
            refused=True,
        )

    def _build_answer(self, picked: Sequence[ScoredSentence]) -> GenerationResult:
        if not self.require_citation:
            body = "\n".join(item.sentence for item in picked)
            return GenerationResult(answer=body, mode=f"{self.mode}-no-citation", citations=[])

        lines: list[str] = []
        used_chunks: list[str] = []
        for index, item in enumerate(picked, start=1):
            lines.append(f"- 结论{index}：{item.sentence} [{item.chunk_id}]")
            if item.chunk_id not in used_chunks:
                used_chunks.append(item.chunk_id)
        lines.append("")
        lines.append("**原文摘录**")
        for chunk_id in used_chunks:
            quotes = [item.sentence for item in picked if item.chunk_id == chunk_id]
            lines.append(f"[{chunk_id}] " + " ".join(quotes))
        return GenerationResult(answer="\n".join(lines), mode=self.mode, citations=used_chunks)

    def generate(self, question: str, hits: Sequence[RetrievedChunk]) -> GenerationResult:
        if self.min_hit_score and hits and hits[0].score < self.min_hit_score:
            return self._refuse(
                f"最佳命中相似度 {hits[0].score:.3f} 低于阈值 {self.min_hit_score}，触发拒答"
            )
        candidates = self._score_sentences(question, hits)
        candidates.sort(key=lambda item: -item.score)
        picked = self._pick(candidates)
        if not picked:
            return self._refuse("检索结果与问题相关性不足，触发拒答")
        return self._build_answer(picked)


class LlmGenerator(BaseGenerator):
    """OpenAI 兼容 /chat/completions；失败时抛错而不是返回半成品答案。"""

    mode = "llm"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 30,
        temperature: float = 0.1,
        require_citation: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.require_citation = require_citation

    def _rules(self) -> str:
        if self.require_citation:
            return (
                "只能依据给定资料回答，不得使用资料以外的知识。"
                "每条结论后必须用方括号标注来源块编号，例如 [c0003]；"
                "若资料不足以回答，只输出「资料不足，无法回答」。"
            )
        return "依据给定资料回答，不要求标注来源。"

    def _request(self, question: str, hits: Sequence[RetrievedChunk]) -> dict[str, Any]:
        prompt = f"{self._rules()}\n\n问题：{question}\n\n资料：\n{_format_context(hits)}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "你是严谨的企业知识库问答助手，回答简洁，拒绝编造。"},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, question: str, hits: Sequence[RetrievedChunk]) -> GenerationResult:
        try:
            data = self._request(question, hits)
            answer = data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("LLM 调用失败：%s: %s", type(exc).__name__, exc)
            raise RuntimeError(f"LLM 调用失败：{type(exc).__name__}: {exc}") from exc
        mode = self.mode if self.require_citation else f"{self.mode}-no-citation"
        return GenerationResult(answer=answer, mode=mode, citations=extract_citations(answer))


def make_generator(
    mode: str = "auto",
    require_citation: bool = True,
    extractive_max_sentences: int = 1,
    embedder: Any = None,
    min_hit_score: float = 0.0,
) -> tuple[BaseGenerator, str]:
    """返回 (generator, 降级说明)；降级说明为空表示按首选模式运行。"""
    if mode == "extractive":
        return (
            ExtractiveGenerator(
                max_sentences=extractive_max_sentences,
                require_citation=require_citation,
                embedder=embedder,
                min_hit_score=min_hit_score,
            ),
            "",
        )

    api_key = os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if mode == "llm" and not api_key:
        raise RuntimeError("RAG_LLM_MODE=llm 需要设置 RAG_LLM_API_KEY / OPENAI_API_KEY")
    if api_key:
        return (
            LlmGenerator(
                api_key=api_key,
                base_url=os.getenv("RAG_LLM_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
                require_citation=require_citation,
            ),
            "",
        )

    note = "降级：未配置 RAG_LLM_API_KEY，使用抽取式生成（答案直接取自原文句子）"
    logger.info(note)
    return (
        ExtractiveGenerator(
            max_sentences=extractive_max_sentences,
            require_citation=require_citation,
            embedder=embedder,
            min_hit_score=min_hit_score,
        ),
        note,
    )
