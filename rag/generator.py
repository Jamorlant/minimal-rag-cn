"""答案生成：可选 LLM（OpenAI 兼容接口），不可用时抽取式兜底。

两种模式都强制（或显式关闭）引用：
- LlmGenerator：提示词要求每个结论后标注 [chunk_id]，资料不足时拒答；
- ExtractiveGenerator：直接返回命中最高的原文句子并附块编号与摘录，零幻觉。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .citations import extract_citations
from .textutil import overlap_score, split_sentences


@dataclass
class GenerationResult:
    answer: str
    mode: str
    note: str = ""
    citations: List[str] = field(default_factory=list)
    refused: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "note": self.note,
            "citations": self.citations,
            "refused": self.refused,
        }


def _format_context(hits: Sequence) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit.chunk_id}]（来源：{hit.doc} · {hit.heading}）\n{hit.text}")
    return "\n\n".join(blocks)


class ExtractiveGenerator:
    mode = "extractive"

    def __init__(
        self,
        max_sentences: int = 1,
        min_score: float = 0.1,
        require_citation: bool = True,
        embedder=None,
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
        # 抽取式只在最相关的一个块内选句：跨块拼接容易把相邻话题的句子凑成答案
        self.max_chunks = max(1, max_chunks)

    def _score_sentences(self, question: str, hits: Sequence) -> List[tuple]:
        """向量余弦为主、字面覆盖为辅：语义相近但没答到问题焦点的句子会被压低。"""
        candidates = []  # (命中块, 句子)
        for hit in list(hits)[: self.max_chunks]:
            for sent in split_sentences(hit.text):
                candidates.append((hit, sent))
        if not candidates:
            return []
        if self.embedder is not None:
            vectors = self.embedder.encode([q[1] for q in candidates])  # type: ignore[union-attr]
            q_vec = self.embedder.encode([question], is_query=True)[0]
            cos = vectors @ q_vec
            scored = []
            for (hit, sent), c in zip(candidates, cos):
                score = float(c) + 0.2 * overlap_score(sent, question) + 0.02 / max(1, hit.rank)
                scored.append((score, hit.chunk_id, sent))
            return scored
        scored = []
        for hit, sent in candidates:
            score = overlap_score(sent, question) + 0.15 * max(0.0, hit.score)
            scored.append((score, hit.chunk_id, sent))
        return scored

    def generate(self, question: str, hits: Sequence) -> GenerationResult:
        if self.min_hit_score and hits and hits[0].score < self.min_hit_score:
            return GenerationResult(
                answer="资料不足，无法回答该问题。",
                mode=self.mode,
                note=f"最佳命中相似度 {hits[0].score:.3f} 低于阈值 {self.min_hit_score}，触发拒答",
                citations=[],
                refused=True,
            )
        candidates = self._score_sentences(question, hits)
        candidates.sort(key=lambda x: -x[0])
        picked = []
        seen: set[str] = set()
        best = candidates[0][0] if candidates else 0.0
        for score, cid, sent in candidates:
            if score < self.min_score:
                continue
            if len(picked) >= self.max_sentences:
                break
            if picked and score < best * self.keep_ratio:
                break  # 次优句明显不如首句时不再凑数，避免答非所问
            key = "".join(sent.split())
            if key in seen:  # 重叠切块会产生重复句，避免同一结论出现两次
                continue
            seen.add(key)
            picked.append((cid, sent, score))

        if not picked:
            return GenerationResult(
                answer="资料不足，无法回答该问题。",
                mode=self.mode,
                note="检索结果与问题相关性不足，触发拒答",
                citations=[],
                refused=True,
            )

        if not self.require_citation:
            # 基线模式：不标注来源，仅拼接抽取到的句子
            body = "\n".join(sent for _, sent, _ in picked)
            return GenerationResult(answer=body, mode=f"{self.mode}-no-citation", citations=[])

        lines = []
        used_chunks: List[str] = []
        for idx, (cid, sent, _) in enumerate(picked, start=1):
            lines.append(f"- 结论{idx}：{sent} [{cid}]")
            if cid not in used_chunks:
                used_chunks.append(cid)
        lines.append("")
        lines.append("**原文摘录**")
        for cid in used_chunks:
            quotes = [sent for c, sent, _ in picked if c == cid]
            lines.append(f"[{cid}] " + " ".join(quotes))
        return GenerationResult(
            answer="\n".join(lines),
            mode=self.mode,
            citations=used_chunks,
        )


class LlmGenerator:
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

    def generate(self, question: str, hits: Sequence) -> GenerationResult:
        if self.require_citation:
            rules = (
                "只能依据给定资料回答，不得使用资料以外的知识。"
                "每条结论后必须用方括号标注来源块编号，例如 [c0003]；"
                "若资料不足以回答，只输出「资料不足，无法回答」。"
            )
        else:
            rules = "依据给定资料回答，不要求标注来源。"
        prompt = f"{rules}\n\n问题：{question}\n\n资料：\n{_format_context(hits)}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "你是严谨的企业知识库问答助手，回答简洁，拒绝编造。"},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"].strip()
            mode = self.mode if self.require_citation else f"{self.mode}-no-citation"
            return GenerationResult(answer=answer, mode=mode, citations=extract_citations(answer))
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            raise RuntimeError(f"LLM 调用失败：{type(exc).__name__}: {exc}") from exc


def make_generator(
    mode: str = "auto",
    require_citation: bool = True,
    extractive_max_sentences: int = 1,
    embedder=None,
    min_hit_score: float = 0.0,
) -> tuple:
    """返回 (generator, 实际模式说明)。LLM 缺少 Key 时自动降级为抽取式。"""
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
        raise RuntimeError("RAG_LLM 模式需要设置 RAG_LLM_API_KEY / OPENAI_API_KEY")
    if api_key:
        gen = LlmGenerator(
            api_key=api_key,
            base_url=os.getenv("RAG_LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            require_citation=require_citation,
        )
        return gen, ""
    gen = ExtractiveGenerator(
        max_sentences=extractive_max_sentences,
        require_citation=require_citation,
        embedder=embedder,
        min_hit_score=min_hit_score,
    )
    return gen, "降级：未配置 RAG_LLM_API_KEY，使用抽取式生成（答案直接取自原文句子）"
