"""文本规范化与中文特征提取工具。"""
from __future__ import annotations

import re
import zlib
from typing import Iterable, List

_PUNCT = re.compile(
    r"[\s\u3000\u00a0，。！？、；：“”‘’（）《》【】…—·,.!?;:'\"()\[\]{}<>/\\|@#$%^&*+=~`_-]+"
)
_SENT_SPLIT = re.compile(r"[。！？；\n]+")


def normalize(text: str) -> str:
    """小写并去除标点空白，得到用于匹配的紧凑文本。"""
    return _PUNCT.sub("", text.lower())


def split_sentences(text: str, min_len: int = 6) -> List[str]:
    """按中文标点切句，过短的碎片与相邻句合并。"""
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    merged: List[str] = []
    for sent in raw:
        if merged and len(merged[-1]) < min_len:
            merged[-1] = merged[-1] + "。" + sent
        else:
            merged.append(sent)
    return [s for s in merged if len(s) >= min_len]


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """字符 n-gram，中文无需分词即可保留词序信息。"""
    compact = normalize(text)
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def ngram_set(text: str, n: int = 2) -> set[str]:
    return set(char_ngrams(text, n))


def containment(claim: str, source: str, n: int = 2) -> float:
    """claim 的 n-gram 有多大比例出现在 source 中，取值 0~1。"""
    claim_grams = ngram_set(claim, n)
    if not claim_grams:
        return 0.0
    return len(claim_grams & ngram_set(source, n)) / len(claim_grams)


def overlap_score(text: str, query: str) -> float:
    """双字覆盖率与单字覆盖率的加权组合，用于抽取式排序。"""
    if not text or not query:
        return 0.0
    big = containment(query, text, 2)
    uni = containment(query, text, 1)
    return 0.7 * big + 0.3 * uni


def support_score(claim: str, source: str) -> float:
    """断言支撑度：双字覆盖率与单字覆盖率的等权组合。

    纯双字覆盖率对改写过于严格，纯单字覆盖率又过于宽松，等权组合能同时容忍轻微改写与识别无关内容。
    """
    if not claim or not source:
        return 0.0
    return 0.5 * containment(claim, source, 2) + 0.5 * containment(claim, source, 1)


def stable_hash(token: str, dim: int) -> int:
    """跨进程稳定的哈希，避免 Python hash 随机种子导致索引不可复现。"""
    return zlib.crc32(token.encode("utf-8")) % dim


def join_texts(texts: Iterable[str]) -> str:
    return " ".join(t for t in texts if t).strip()
