"""引用解析与断言支撑校验。

校验口径（README 与 docs 同步，避免「引用编号存在即正确」的形式化验收）：
1. 答案按行、再按中文句末标点切分为独立断言；
2. 每条断言与被引用块求「双字覆盖率 0.5 + 单字覆盖率 0.5」的混合分；
3. 混合分 >= 阈值（默认 0.55）记为「已支撑」，否则「未支撑」；
4. 未标注引用的断言一律未支撑；编号不在召回集合中时单独记为越界引用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .textutil import support_score

CITATION_RE = re.compile(r"\[(c\d{4,})\]")
_QUOTE_LABELS = {"原文摘录", "来源", "依据"}
_OTHER_LABELS = {"引用校验", "参考"}


@dataclass
class ClaimCheck:
    claim: str
    cited_ids: list[str]
    best_score: float
    supported: bool


@dataclass
class VerificationResult:
    claims: list[ClaimCheck] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for claim in self.claims if claim.supported)

    @property
    def support_rate(self) -> float:
        return self.supported / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total,
            "supported": self.supported,
            "unsupported": self.total - self.supported,
            "support_rate": round(self.support_rate, 4),
            "unknown_ids": self.unknown_ids,
            "claims": [
                {
                    "claim": claim.claim,
                    "cited_ids": claim.cited_ids,
                    "best_score": round(claim.best_score, 4),
                    "supported": claim.supported,
                }
                for claim in self.claims
            ],
        }


def extract_citations(text: str) -> list[str]:
    citations: list[str] = []
    for chunk_id in CITATION_RE.findall(text):
        if chunk_id not in citations:
            citations.append(chunk_id)
    return citations


def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def claim_pairs(answer: str) -> list[tuple[str, list[str]]]:
    """切分为 (断言, 该行引用编号)，保留断言与引用的对应关系。

    「原文摘录」是展示证据而非结论，跳过该段，避免把同一句话重复计入断言。
    """
    pairs: list[tuple[str, list[str]]] = []
    in_quote_section = False
    for line in answer.splitlines():
        stripped = _strip_citations(line).strip().strip("*# ").strip()
        if stripped in _QUOTE_LABELS:
            in_quote_section = True
            continue
        if stripped in _OTHER_LABELS:
            in_quote_section = False
            continue
        if in_quote_section:
            continue
        cited = extract_citations(line)
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.、])\s*", "", stripped)
        cleaned = re.sub(
            r"^(结论|原文摘录|依据|答案)[一二三四五六七八九十\d]*[：:]\s*", "", cleaned
        )
        if not cleaned or len(cleaned) < 6:
            continue
        for part in re.split(r"[。！？；]", cleaned):
            part = part.strip("*# ")
            if len(part) >= 6 and part not in _QUOTE_LABELS:
                pairs.append((part, cited))
    return pairs


def verify_answer(
    answer: str,
    cited_texts: dict[str, str],
    threshold: float = 0.55,
) -> VerificationResult:
    """逐条断言校验能否在被引用的块中找到支撑；未知编号单独记录、不吞掉。"""
    result = VerificationResult()
    mentioned = extract_citations(answer)
    result.unknown_ids = [chunk_id for chunk_id in mentioned if chunk_id not in cited_texts]
    fallback_ids = [chunk_id for chunk_id in mentioned if chunk_id in cited_texts]

    for claim, ids in claim_pairs(answer):
        effective = [chunk_id for chunk_id in (ids or fallback_ids) if chunk_id in cited_texts]
        best_score = 0.0
        best_id = ""
        for chunk_id in effective:
            score = support_score(claim, cited_texts[chunk_id])
            if score > best_score:
                best_score, best_id = score, chunk_id
        result.claims.append(
            ClaimCheck(
                claim=claim,
                cited_ids=[best_id] if best_id else [],
                best_score=best_score,
                supported=best_score >= threshold,
            )
        )
    return result
