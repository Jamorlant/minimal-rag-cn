"""索引构建与加载。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .chunker import build_chunks, chunk_embed_text, load_paragraphs
from .config import Config
from .embeddings import BaseEmbedder, EmbedderFactory, NgramTfidfEmbedder
from .store import BaseStore, StoreFactory


def build_index(cfg: Config) -> dict:
    """切块 -> 向量化 -> 写入向量库，返回索引元信息。"""
    paragraphs = load_paragraphs(cfg.paragraphs_path)
    chunks = build_chunks(paragraphs, chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    if not chunks:
        raise RuntimeError("切块结果为空，请检查语料")

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    with cfg.chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    embedder = EmbedderFactory(cfg.embed_mode, model_name=cfg.model_name, dim=cfg.embed_dim)
    texts = [chunk_embed_text(c) for c in chunks]
    embedder.fit(texts)
    embeddings = embedder.encode(texts)

    store, store_note = StoreFactory.create(cfg.store_mode, cfg.index_dir / "vectors", reset=True)
    store.add(chunks, embeddings)

    if isinstance(embedder, NgramTfidfEmbedder):
        embedder.save_idf(cfg.index_dir / "idf.npy")

    meta = {
        "chunk_size": cfg.chunk_size,
        "overlap": cfg.overlap,
        "num_paragraphs": len(paragraphs),
        "num_chunks": len(chunks),
        "embedder": {
            "kind": type(embedder).__name__,
            "name": embedder.name,
            "dim": embedder.dim,
            "degraded": embedder.degraded,
            "note": embedder.note,
        },
        "store": {"kind": store.kind, "degraded": store.degraded, "note": store_note},
        "total_chars": sum(len(p.text) for p in paragraphs),
    }
    cfg.index_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_index(cfg: Config) -> Tuple[BaseEmbedder, BaseStore, dict]:
    if not cfg.index_meta_path.exists():
        raise FileNotFoundError("索引不存在，请先运行 python cli.py build")
    meta = json.loads(cfg.index_meta_path.read_text(encoding="utf-8"))

    if meta["embedder"]["kind"] == "NgramTfidfEmbedder":
        embedder = NgramTfidfEmbedder(dim=int(meta["embedder"]["dim"]))
        embedder.load_idf(cfg.index_dir / "idf.npy")
        embedder.note = meta["embedder"]["note"]
    else:
        embedder = EmbedderFactory("st", model_name=meta["embedder"]["name"])

    store_kind = "numpy" if meta["store"]["kind"] == "numpy" else "chroma"
    store, _ = StoreFactory.create(store_kind, cfg.index_dir / "vectors", reset=False)
    return embedder, store, meta


def load_chunks(cfg: Config) -> List[dict]:
    rows = []
    with cfg.chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
