"""引用解析与断言支撑校验。

校验口径（同时写入 README，避免「引用编号存在即正确」的形式化验收）：
1. 答案按行、再按中文句末标点切分为独立断言；
2. 每条断言取其字符二元组集合，与被引用块文本的二元组集合求覆盖率；
3. 覆盖率 >= 阈值（默认 0.55）记为「已支撑」，否则「未支撑」；
4. 未标注引用的断言一律记为「未支撑」；引用编号不在召回集合中时单独记录为越界引用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .textutil import support_score

CITATION_RE = re.compile(r"\[(c\d{4,})\]")


@dataclass
class ClaimCheck:
    claim: str
    cited_ids: List[str]
    best_score: float
    supported: bool


@dataclass
class VerificationResult:
    claims: List[ClaimCheck] = field(default_factory=list)
    unknown_ids: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.supported)

    @property
    def support_rate(self) -> float:
        return self.supported / self.total if self.total else 0.0

    @property
    def all_supported(self) -> bool:
        return self.total > 0 and self.supported == self.total

    def to_dict(self) -> dict:
        return {
            "total_claims": self.total,
            "supported": self.supported,
            "unsupported": self.total - self.supported,
            "support_rate": round(self.support_rate, 4),
            "unknown_ids": self.unknown_ids,
            "claims": [
                {
                    "claim": c.claim,
                    "cited_ids": c.cited_ids,
                    "best_score": round(c.best_score, 4),
                    "supported": c.supported,
                }
                for c in self.claims
            ],
        }


def extract_citations(text: str) -> List[str]:
    seen: List[str] = []
    for cid in CITATION_RE.findall(text):
        if cid not in seen:
            seen.append(cid)
    return seen


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def claim_pairs(answer: str) -> List[Tuple[str, List[str]]]:
    """切分为 (断言, 该行引用编号) 列表，保留断言与引用的对应关系。

    「原文摘录」属于展示证据而非答案断言，校验时跳过该段，避免把摘录重复计入断言。
    """
    pairs: List[Tuple[str, List[str]]] = []
    labels = {"原文摘录", "引用校验", "参考", "来源", "依据"}
    in_quote_section = False
    for line in answer.splitlines():
        stripped = strip_citations(line).strip().strip("*# ").strip()
        if stripped in labels:
            in_quote_section = stripped in {"原文摘录", "来源", "依据"}
            continue
        if in_quote_section:
            continue
        ids = extract_citations(line)
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.、])\s*", "", stripped)
        cleaned = re.sub(r"^(结论|原文摘录|依据|答案)[一二三四五六七八九十\d]*[：:]\s*", "", cleaned)
        if not cleaned or len(cleaned) < 6:
            continue
        for part in re.split(r"[。！？；]", cleaned):
            part = part.strip("*# ")
            if len(part) >= 6 and part not in labels:
                pairs.append((part, ids))
    return pairs


def verify_answer(
    answer: str,
    cited_texts: Dict[str, str],
    threshold: float = 0.55,
) -> VerificationResult:
    """逐条断言校验其能否在被引用的块中找到支撑。"""
    result = VerificationResult()
    all_ids = [cid for cid in extract_citations(answer) if cid in cited_texts]
    result.unknown_ids = [cid for cid in extract_citations(answer) if cid not in cited_texts]

    for claim, ids in claim_pairs(answer):
        effective = [cid for cid in (ids or all_ids) if cid in cited_texts]
        best = 0.0
        best_id = ""
        for cid in effective:
            score = support_score(claim, cited_texts[cid])
            if score > best:
                best, best_id = score, cid
        result.claims.append(
            ClaimCheck(
                claim=claim,
                cited_ids=[best_id] if best_id else [],
                best_score=best,
                supported=best >= threshold,
            )
        )
    return result


def verify_with_chunks(
    answer: str,
    chunks: Sequence,
    threshold: float = 0.55,
) -> VerificationResult:
    """便捷包装：chunks 为 RetrievedChunk 列表。"""
    return verify_answer(answer, {c.chunk_id: c.text for c in chunks}, threshold=threshold)
