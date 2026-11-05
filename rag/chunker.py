"""语料加载与切块。

切块策略：
1. 以段落为最小语义单元，按 chunk_size（字符数）贪心打包相邻段落；
2. 段落自身超过 chunk_size 时，按句子边界二次切分为字符窗口，保证不截断句子；
3. overlap 通过「把上一块末尾若干段落复制到下一块开头」实现，块间保留上下文重叠。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from .textutil import split_sentences


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
    para_ids: List[str] = field(default_factory=list)
    start_para: int = 0
    end_para: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc": self.doc,
            "heading": self.heading,
            "text": self.text,
            "para_ids": self.para_ids,
            "start_para": self.start_para,
            "end_para": self.end_para,
        }


def load_paragraphs(path: Path) -> List[Paragraph]:
    if not path.exists():
        raise FileNotFoundError(f"段落文件不存在：{path}，请先运行 python scripts/build_corpus.py")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Paragraph(**json.loads(line)))
    return rows


def _split_long_paragraph(text: str, size: int, overlap: int) -> List[str]:
    """按句子边界把超长段落切成窗口，窗口间保留字符级重叠。"""
    if len(text) <= size:
        return [text]
    sentences = split_sentences(text, min_len=4) or [text]
    windows: List[str] = []
    buf = ""
    for sent in sentences:
        piece = sent + "。"
        if buf and len(buf) + len(piece) > size:
            windows.append(buf)
            buf = buf[-overlap:] if overlap > 0 else ""
        if len(piece) > size:
            step = max(1, size - overlap)
            for start in range(0, len(piece), step):
                windows.append(piece[start : start + size])
            buf = ""
            continue
        buf += piece
    if buf.strip():
        windows.append(buf)
    return windows


def build_chunks(
    paragraphs: Iterable[Paragraph],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    paragraphs = list(paragraphs)
    chunks: List[Chunk] = []
    i = 0
    while i < len(paragraphs):
        # 单段超长：独立按句切分，不复用打包逻辑
        if len(paragraphs[i].text) > chunk_size:
            para = paragraphs[i]
            for part in _split_long_paragraph(para.text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        id=f"c{len(chunks):04d}",
                        doc=para.doc,
                        heading=para.heading,
                        text=part,
                        para_ids=[para.id],
                        start_para=i,
                        end_para=i,
                    )
                )
            i += 1
            continue

        buf: List[Paragraph] = []
        length = 0
        j = i
        while j < len(paragraphs) and len(paragraphs[j].text) <= chunk_size:
            # 标题或文档变化时强制断开，保证一个块只属于一个语义小节
            if buf and (
                paragraphs[j].doc != buf[0].doc or paragraphs[j].heading != buf[0].heading
            ):
                break
            add = len(paragraphs[j].text) + (1 if buf else 0)
            if buf and length + add > chunk_size:
                break
            buf.append(paragraphs[j])
            length += add
            j += 1

        body = "\n".join(p.text for p in buf)
        chunks.append(
            Chunk(
                id=f"c{len(chunks):04d}",
                doc=buf[0].doc,
                heading=buf[0].heading,
                text=body,
                para_ids=[p.id for p in buf],
                start_para=i,
                end_para=j - 1,
            )
        )

        # 计算重叠：从块尾向前取若干段，总长不超过 overlap
        carry_len = 0
        carry_count = 0
        for p in reversed(buf):
            if carry_len + len(p.text) > overlap:
                break
            carry_len += len(p.text) + 1
            carry_count += 1
        # 若重叠段覆盖整块，则不再回退，避免原地打转
        if 0 < carry_count < len(buf):
            i = j - carry_count
        else:
            i = j
    return chunks


def chunk_embed_text(chunk: Chunk) -> str:
    """送给向量模型的文本：拼接标题以增强主题信号。"""
    return f"{chunk.doc} · {chunk.heading}：{chunk.text}"
