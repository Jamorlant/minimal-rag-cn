"""运行配置：命令行参数优先，其次环境变量，最后默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Config:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 280)
    overlap: int = _env_int("RAG_CHUNK_OVERLAP", 56)
    top_k: int = _env_int("RAG_TOP_K", 3)
    rerank_weight: float = _env_float("RAG_RERANK_WEIGHT", 0.5)  # 稠密分数占比，其余为字符 n-gram 字面分
    min_hit_score: float = _env_float("RAG_MIN_HIT_SCORE", 0.25)  # 最佳命中低于该值直接拒答
    embed_mode: str = os.getenv("RAG_EMBED_MODE", "auto")  # auto | st | ngram
    embed_dim: int = _env_int("RAG_EMBED_DIM", 2048)
    store_mode: str = os.getenv("RAG_STORE", "auto")  # auto | chroma | numpy
    llm_mode: str = os.getenv("RAG_LLM_MODE", "auto")  # auto | llm | extractive
    support_threshold: float = _env_float("RAG_SUPPORT_THRESHOLD", 0.55)
    model_name: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def paragraphs_path(self) -> Path:
        return self.data_dir / "paragraphs.jsonl"

    @property
    def qa_path(self) -> Path:
        return self.data_dir / "eval_qa.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def index_meta_path(self) -> Path:
        return self.index_dir / "meta.json"
