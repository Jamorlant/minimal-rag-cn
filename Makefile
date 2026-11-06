.PHONY: help install install-offline install-dev build demo eval test lint fmt clean

PY ?= python

help:
	@echo "make install        安装全部依赖（含 BGE / ChromaDB）"
	@echo "make install-offline 只装 numpy（走离线降级链路）"
	@echo "make install-dev    安装 lint 工具（ruff）"
	@echo "make build          构建切块与索引"
	@echo "make demo           一键演示（写入 outputs/）"
	@echo "make eval           30 题评测（写入 outputs/）"
	@echo "make test           单元测试"
	@echo "make lint           ruff 静态检查"
	@echo "make fmt            ruff 自动修复"
	@echo "make clean          清理索引与缓存"

install:
	$(PY) -m pip install -r requirements.txt -r requirements-optional.txt

install-offline:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

build:
	$(PY) cli.py build

demo:
	$(PY) cli.py demo

eval:
	$(PY) cli.py eval

test:
	$(PY) -m unittest discover -s tests -v

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff check --fix .

clean:
	rm -rf index outputs/*.json __pycache__ .ruff_cache .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
