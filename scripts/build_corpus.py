"""从 data/corpus/*.md 解析段落，生成 data/paragraphs.jsonl。

语料为本项目自行撰写的通用中文技术笔记，不含任何真实企业资料或个人信息。
段落 ID 按文件顺序与段落顺序稳定生成，重建后保持一致。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
OUT_PATH = ROOT / "data" / "paragraphs.jsonl"

HeadingParagraph = tuple[str, str]


def parse_file(path: Path) -> tuple[str, list[HeadingParagraph]]:
    """解析单篇文档：`#` 为文档名，`##` 为小节，空行分段。"""
    doc = path.stem
    heading = "正文"
    paragraphs: list[HeadingParagraph] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = re.sub(r"\s+", " ", "".join(buffer)).strip()
        if text:
            paragraphs.append((heading, text))
        buffer.clear()

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            doc = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line.strip())
    flush()
    return doc, paragraphs


def main() -> int:
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print(f"{CORPUS_DIR} 下没有 md 文件", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for path in files:
        doc, paragraphs = parse_file(path)
        for heading, text in paragraphs:
            rows.append(
                {
                    "id": f"p{len(rows) + 1:04d}",
                    "doc": doc,
                    "heading": heading,
                    "text": text,
                }
            )

    OUT_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    chars = sum(len(row["text"]) for row in rows)
    print(f"段落数：{len(rows)}，字符数：{chars}，约 {chars / 750:.1f} 页（按 750 字/页）")
    print(f"文档数：{len({row['doc'] for row in rows})}")
    print(f"已写入 {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
