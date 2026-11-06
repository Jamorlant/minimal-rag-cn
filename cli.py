"""命令行入口：build / query / demo / eval。

CLI 只负责参数解析与面向用户的输出，业务逻辑全部在 rag 包内。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from rag.config import Config
from rag.evaluate import run_evaluation, save_eval_outputs
from rag.index import build_index
from rag.logging_utils import get_logger
from rag.pipeline import RagAnswer, RagPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger(__name__)

DEMO_QUESTIONS: list[str] = [
    "快速排序的最坏时间复杂度是多少，什么情况下会退化？",
    "pandas 里 merge 的 how 参数支持哪几种连接方式？",
    "BGE 中文小模型的输出向量维度是多少，查询侧应该怎么处理？",
    "公司差旅报销的住宿标准是多少钱一晚？",
]


def _ensure_index(cfg: Config, command: str) -> None:
    if cfg.index_meta_path.exists():
        return
    print(f"[{command}] 未发现索引，先自动构建……")
    cmd_build(cfg)


def cmd_build(cfg: Config) -> None:
    meta = build_index(cfg)
    print("[build] 完成")
    print(
        f"  段落数：{meta['num_paragraphs']}，切块数：{meta['num_chunks']}，"
        f"字符数：{meta['total_chars']}"
    )
    print(f"  向量模型：{meta['embedder']['name']}")
    if meta["embedder"]["degraded"]:
        print(f"  ! {meta['embedder']['note']}")
    print(f"  向量库：{meta['store']['kind']}")
    if meta["store"]["degraded"]:
        print(f"  ! {meta['store']['note']}")


def _print_answer(result: RagAnswer) -> None:
    print(f"\n问题：{result.question}")
    print("召回结果：")
    for hit in result.hits:
        snippet = hit.text.replace("\n", " ")[:90]
        print(
            f"  [{hit.rank}] {hit.chunk_id} score={hit.score:.3f} "
            f"{hit.doc}/{hit.heading} :: {snippet}"
        )
    print(f"\n回答（{result.generation.mode}）：\n{result.generation.answer}")
    if result.generation.refused:
        print("\n引用校验：系统按「资料不足」拒答，未生成可核验断言")
    else:
        verification = result.verification
        print(
            f"\n引用校验：{verification.supported}/{verification.total} 条断言已支撑，"
            f"支撑率 {verification.support_rate:.1%}"
        )
    if result.verification.unknown_ids:
        print(f"  越界引用：{result.verification.unknown_ids}")
    for note in result.notes:
        print(f"  提示：{note}")


def cmd_query(cfg: Config, question: str, k: int, require_citation: bool) -> None:
    _ensure_index(cfg, "query")
    pipeline = RagPipeline(cfg)
    _print_answer(pipeline.ask(question, k=k, require_citation=require_citation))


def _demo_header(cfg: Config, pipeline: RagPipeline, results: Sequence[RagAnswer]) -> list[str]:
    return [
        "# 最小 RAG 演示结果",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 索引配置：chunk_size={cfg.chunk_size}，overlap={cfg.overlap}，top_k={cfg.top_k}",
        (
            f"- 向量模型：{pipeline.meta['embedder']['name']}"
            f"（降级={pipeline.meta['embedder']['degraded']}）"
        ),
        f"- 向量库：{pipeline.meta['store']['kind']}（降级={pipeline.meta['store']['degraded']}）",
        f"- 生成模式：{results[0].generation.mode}" if results else "",
        "",
    ]


def _demo_hits(result: RagAnswer) -> list[str]:
    lines = ["| 排名 | chunk | 相似度 | 来源 |", "| --- | --- | --- | --- |"]
    for hit in result.hits:
        lines.append(
            f"| {hit.rank} | {hit.chunk_id} | {hit.score:.3f} | {hit.doc} · {hit.heading} |"
        )
    return lines


def _demo_verification(result: RagAnswer) -> list[str]:
    verification = result.verification
    if result.generation.refused:
        return ["**引用校验**：未生成断言，系统按「资料不足」拒答", ""]
    lines = [
        f"**引用校验**：{verification.supported}/{verification.total} 条断言已支撑，"
        f"支撑率 {verification.support_rate:.1%}"
        + (f"；越界引用 {verification.unknown_ids}" if verification.unknown_ids else ""),
        "",
        "| 断言 | 引用块 | 覆盖率 | 结论 |",
        "| --- | --- | --- | --- |",
    ]
    for claim in verification.claims:
        lines.append(
            f"| {claim.claim} | {','.join(claim.cited_ids) or '-'} | "
            f"{claim.best_score:.2f} | {'已支撑' if claim.supported else '未支撑'} |"
        )
    lines.append("")
    return lines


def _build_demo_markdown(cfg: Config, pipeline: RagPipeline, results: Sequence[RagAnswer]) -> str:
    lines = _demo_header(cfg, pipeline, results)
    for index, result in enumerate(results, start=1):
        lines.extend([f"## 示例 {index}：{result.question}", ""])
        lines.extend(_demo_hits(result))
        lines.extend(["", "**回答**", "", result.generation.answer, ""])
        lines.extend(_demo_verification(result))
        if result.notes:
            lines.extend(["提示：" + "；".join(result.notes), ""])
    return "\n".join(lines)


def cmd_demo(cfg: Config) -> None:
    _ensure_index(cfg, "demo")
    pipeline = RagPipeline(cfg)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    results = [pipeline.ask(question, k=cfg.top_k) for question in DEMO_QUESTIONS]

    (cfg.outputs_dir / "demo.md").write_text(
        _build_demo_markdown(cfg, pipeline, results), encoding="utf-8"
    )
    payload: list[dict[str, Any]] = [result.to_dict() for result in results]
    (cfg.outputs_dir / "demo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[demo] 已写出 outputs/demo.md 与 outputs/demo.json（{len(results)} 个示例）")


def cmd_eval(cfg: Config, limit: int) -> None:
    _ensure_index(cfg, "eval")
    pipeline = RagPipeline(cfg)
    results = run_evaluation(pipeline, cfg, limit=limit)
    path = save_eval_outputs(results, cfg, pipeline.meta)
    print(f"[eval] 评测题数：{results[0]['num_questions']}")
    for result in results:
        print(
            f"  {result['label']}：命中率@{result['k']}={result['hit_rate_at_k']:.2%}，"
            f"答案正确率={result['answer_accuracy']:.2%}，"
            f"引用命中率={result['citation_accuracy']:.2%}，"
            f"支撑率={result['support_rate']:.2%}"
        )
    print(f"[eval] 报告已写入 {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小中文 RAG 项目")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="构建切块与向量索引")

    query = sub.add_parser("query", help="检索并回答单个问题")
    query.add_argument("question")
    query.add_argument("--no-citation", action="store_true", help="关闭引用约束")

    sub.add_parser("demo", help="一键跑通示例并写入 outputs/")
    eval_parser = sub.add_parser("eval", help="运行 30 题评测并生成报告")
    eval_parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.overlap = args.overlap
    if args.k:
        cfg.top_k = args.k

    try:
        if args.command == "build":
            cmd_build(cfg)
        elif args.command == "query":
            cmd_query(cfg, args.question, cfg.top_k, not args.no_citation)
        elif args.command == "demo":
            cmd_demo(cfg)
        elif args.command == "eval":
            cmd_eval(cfg, args.limit)
        else:
            parser.print_help()
            return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        print(f"[error] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
