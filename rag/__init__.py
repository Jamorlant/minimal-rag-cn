"""最小中文 RAG 项目核心包。

模块分层：config（配置）→ chunker / embeddings / store（基础能力）
→ index（索引）→ pipeline（编排）→ generator / citations（生成与校验）
→ evaluate（评测）。
"""

__all__ = [
    "config",
    "chunker",
    "embeddings",
    "store",
    "index",
    "pipeline",
    "generator",
    "citations",
    "evaluate",
    "textutil",
    "logging_utils",
]
