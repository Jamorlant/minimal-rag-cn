"""语料加载与切块。

切块策略（中文技术文档的语义单位是段落，不按固定字符硬切）：
1. 以段落为最小单元，贪心打包相邻段落，总长不超过 chunk_size；
2. 同一块内必须同文档、同 `##` 标题，避免把两个小节的句子混进同一块；
3. 段落自身超长时按句边界二次切分，窗口间保留 overlap 字符；
4. 块间重叠只回退「能完整放下的尾部段落」，回退段覆盖整块时不再回退（防死循环）。
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .textutil import split_sentences

logger = get_logger(__name__)


@dataclass
class Paragraph:
    id: str
    doc: str
    heading: str
    text: str


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    text: str
    para_ids: list[str] = field(default_factory=list)
    start_para: int = 0
    end_para: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc": self.doc,
            "heading": self.heading,
            "text": self.text,
            "para_ids": self.para_ids,
            "start_para": self.start_para,
            "end_para": self.end_para,
        }


def load_paragraphs(path: Path) -> list[Paragraph]:
    if not path.exists():
        raise FileNotFoundError(f"段落文件不存在：{path}，请先运行 python scripts/build_corpus.py")
    paragraphs: list[Paragraph] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                paragraphs.append(Paragraph(**json.loads(line)))
    return paragraphs


def _split_long_paragraph(text: str, size: int, overlap: int) -> list[str]:
    """按句子边界把超长段落切成窗口，窗口间保留字符级重叠。"""
    if len(text) <= size:
        return [text]
    sentences = split_sentences(text, min_len=4) or [text]
    windows: list[str] = []
    buffer = ""
    for sentence in sentences:
        piece = sentence + "。"
        if buffer and len(buffer) + len(piece) > size:
            windows.append(buffer)
            buffer = buffer[-overlap:] if overlap > 0 else ""
        if len(piece) > size:
            step = max(1, size - overlap)
            for start in range(0, len(piece), step):
                windows.append(piece[start : start + size])
            buffer = ""
            continue
        buffer += piece
    if buffer.strip():
        windows.append(buffer)
    return windows


def _collect_window(paragraphs: Sequence[Paragraph], start: int, chunk_size: int) -> list[Paragraph]:
    """从 start 开始尽量多地收集同文档同标题的段落。"""
    window: list[Paragraph] = []
    length = 0
    index = start
    while index < len(paragraphs) and len(paragraphs[index].text) <= chunk_size:
        current = paragraphs[index]
        if window and (current.doc != window[0].doc or current.heading != window[0].heading):
            break
        extra = len(current.text) + (1 if window else 0)
        if window and length + extra > chunk_size:
            break
        window.append(current)
        length += extra
        index += 1
    return window


def _make_chunk(window: Sequence[Paragraph], start: int, chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc=window[0].doc,
        heading=window[0].heading,
        text="\n".join(paragraph.text for paragraph in window),
        para_ids=[paragraph.id for paragraph in window],
        start_para=start,
        end_para=start + len(window) - 1,
    )


def _overlap_tail(window: Sequence[Paragraph], overlap: int) -> int:
    """从块尾向前数，能完整放进 overlap 字符预算的段落数。"""
    carry_len = 0
    carry_count = 0
    for paragraph in reversed(window):
        if carry_len + len(paragraph.text) > overlap:
            break
        carry_len += len(paragraph.text) + 1
        carry_count += 1
    return carry_count


def build_chunks(
    paragraphs: Sequence[Paragraph],
    chunk_size: int = 280,
    overlap: int = 56,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须小于 chunk_size 且不小于 0")

    items = list(paragraphs)
    chunks: list[Chunk] = []
    index = 0
    while index < len(items):
        paragraph = items[index]
        if len(paragraph.text) > chunk_size:
            # 单段超长：独立按句切分，不复用打包逻辑
            for part in _split_long_paragraph(paragraph.text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        id=f"c{len(chunks):04d}",
                        doc=paragraph.doc,
                        heading=paragraph.heading,
                        text=part,
                        para_ids=[paragraph.id],
                        start_para=index,
                        end_para=index,
                    )
                )
            index += 1
            continue

        window = _collect_window(items, index, chunk_size)
        chunks.append(_make_chunk(window, index, f"c{len(chunks):04d}"))
        carry = _overlap_tail(window, overlap)
        index = index + len(window) - carry if 0 < carry < len(window) else index + len(window)

    logger.debug("切块完成：%d 段落 → %d 块", len(items), len(chunks))
    return chunks


def chunk_embed_text(chunk: Chunk) -> str:
    """送给向量模型的文本：拼接标题以增强主题信号。"""
    return f"{chunk.doc} · {chunk.heading}：{chunk.text}"
