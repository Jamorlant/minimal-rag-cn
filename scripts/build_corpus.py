"""从 data/corpus/*.md 解析段落，生成 data/paragraphs.jsonl。

语料为本项目自行撰写的通用中文技术笔记，不含任何真实企业资料或个人信息，
可自由用于演示与评测。段落 ID 按文件顺序与段落顺序稳定生成。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
OUT_PATH = ROOT / "data" / "paragraphs.jsonl"


def parse_file(path: Path):
    doc = path.stem
    heading = "正文"
    paragraphs = []
    buf: list[str] = []

    def flush():
        if buf:
            text = re.sub(r"\s+", " ", "".join(buf)).strip()
            if text:
                paragraphs.append((heading, text))
            buf.clear()

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
        buf.append(line.strip())
    flush()
    return doc, paragraphs


def main() -> int:
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print("data/corpus 下没有 md 文件", file=sys.stderr)
        return 1

    rows = []
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
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    chars = sum(len(r["text"]) for r in rows)
    pages = chars / 750
    print(f"段落数：{len(rows)}，字符数：{chars}，约 {pages:.1f} 页（按 750 字/页）")
    docs = {r["doc"] for r in rows}
    print(f"文档数：{len(docs)} -> {sorted(docs)}")
    print(f"已写入 {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
