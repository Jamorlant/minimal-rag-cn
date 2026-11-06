"""统一日志配置。

库代码只写日志，不直接 print；CLI 负责面向用户的输出。
级别由 RAG_LOG_LEVEL 控制，默认 INFO，异常路径用 WARNING/ERROR。
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """幂等初始化根日志：只挂一个 stderr handler，避免重复输出。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.getenv("RAG_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved, logging.INFO))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
