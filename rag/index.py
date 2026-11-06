"""索引构建与加载。

索引目录结构（不入 git）：
    index/chunks.jsonl      切块结果（供重排与展示）
    index/meta.json         版本、切块参数、向量器与向量库状态
    index/idf.npy           仅离线向量器需要
    index/vectors/          ChromaDB 或 numpy 向量
"""
from __future__ import annotations

import json
from typing import Any

from .chunker import build_chunks, chunk_embed_text, load_paragraphs
from .config import Config
from .embeddings import BaseEmbedder, EmbedderFactory, NgramTfidfEmbedder, describe
from .logging_utils import get_logger
from .store import BaseStore, StoreFactory

logger = get_logger(__name__)
INDEX_VERSION = 1


def build_index(cfg: Config) -> dict[str, Any]:
    paragraphs = load_paragraphs(cfg.paragraphs_path)
    chunks = build_chunks(paragraphs, chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    if not chunks:
        raise RuntimeError("切块结果为空，请检查 data/corpus 下的语料")

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    with cfg.chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    embedder = EmbedderFactory.create(cfg.embed_mode, model_name=cfg.model_name, dim=cfg.embed_dim)
    texts = [chunk_embed_text(chunk) for chunk in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    store, store_note = StoreFactory.create(cfg.store_mode, cfg.index_dir / "vectors", reset=True)
    store.add(chunks, vectors)

    if isinstance(embedder, NgramTfidfEmbedder):
        embedder.save_idf(cfg.index_dir / "idf.npy")

    meta: dict[str, Any] = {
        "version": INDEX_VERSION,
        "chunk_size": cfg.chunk_size,
        "overlap": cfg.overlap,
        "num_paragraphs": len(paragraphs),
        "num_chunks": len(chunks),
        "total_chars": sum(len(paragraph.text) for paragraph in paragraphs),
        "embedder": describe(embedder),
        "store": {"kind": store.kind, "degraded": store.degraded, "note": store_note},
    }
    cfg.index_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "索引完成：%d 段落 → %d 块，向量器=%s，向量库=%s",
        len(paragraphs),
        len(chunks),
        embedder.name,
        store.kind,
    )
    return meta


def load_index(cfg: Config) -> tuple[BaseEmbedder, BaseStore, dict[str, Any]]:
    if not cfg.index_meta_path.exists():
        raise FileNotFoundError("索引不存在，请先运行 python cli.py build")
    meta: dict[str, Any] = json.loads(cfg.index_meta_path.read_text(encoding="utf-8"))

    if meta["embedder"]["kind"] == "NgramTfidfEmbedder":
        embedder: BaseEmbedder = NgramTfidfEmbedder(dim=int(meta["embedder"]["dim"]))
        assert isinstance(embedder, NgramTfidfEmbedder)
        embedder.load_idf(cfg.index_dir / "idf.npy")
        embedder.note = meta["embedder"]["note"]
    else:
        # 索引里记录了实际使用的模型；离线时只读本地缓存
        embedder = EmbedderFactory.create("st", model_name=meta["embedder"]["name"])

    store_kind = "numpy" if meta["store"]["kind"] == "numpy" else "chroma"
    store, _ = StoreFactory.create(store_kind, cfg.index_dir / "vectors", reset=False)
    return embedder, store, meta


def load_chunks(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.chunks_path.exists():
        raise FileNotFoundError(f"切块文件不存在：{cfg.chunks_path}，请先运行 python cli.py build")
    rows: list[dict[str, Any]] = []
    with cfg.chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
