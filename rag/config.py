"""运行配置：命令行参数优先，其次环境变量与 .env，最后默认值。

敏感项（LLM Key）只从环境读取，仓库内不保存任何真实值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """最小 .env 解析：KEY=VALUE，忽略注释与空行；已存在的环境变量不覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值：{raw!r}") from exc


load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    root: Path = field(default_factory=lambda: PROJECT_ROOT)
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 280)
    overlap: int = _env_int("RAG_CHUNK_OVERLAP", 56)
    top_k: int = _env_int("RAG_TOP_K", 3)
    # 稠密分在混合重排中的权重，其余给字符 n-gram 字面分
    rerank_weight: float = _env_float("RAG_RERANK_WEIGHT", 0.5)
    # 最佳命中分低于该值直接拒答，避免强行作答
    min_hit_score: float = _env_float("RAG_MIN_HIT_SCORE", 0.25)
    embed_mode: str = os.getenv("RAG_EMBED_MODE", "auto")  # auto | st | ngram
    embed_dim: int = _env_int("RAG_EMBED_DIM", 2048)
    store_mode: str = os.getenv("RAG_STORE", "auto")  # auto | chroma | numpy
    llm_mode: str = os.getenv("RAG_LLM_MODE", "auto")  # auto | llm | extractive
    support_threshold: float = _env_float("RAG_SUPPORT_THRESHOLD", 0.55)
    model_name: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size 且不小于 0")
        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if not 0.0 <= self.rerank_weight <= 1.0:
            raise ValueError("rerank_weight 必须在 0~1 之间")
        if not 0.0 <= self.support_threshold <= 1.0:
            raise ValueError("support_threshold 必须在 0~1 之间")

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
