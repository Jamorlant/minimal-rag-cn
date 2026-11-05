"""命令行入口：build / query / demo / eval。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rag.config import Config
from rag.index import build_index
from rag.pipeline import RagPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cmd_build(cfg: Config) -> None:
    meta = build_index(cfg)
    print("[build] 完成")
    print(f"  段落数：{meta['num_paragraphs']}，切块数：{meta['num_chunks']}，字符数：{meta['total_chars']}")
    print(f"  向量模型：{meta['embedder']['name']}")
    if meta["embedder"]["degraded"]:
        print(f"  ! {meta['embedder']['note']}")
    print(f"  向量库：{meta['store']['kind']}")
    if meta["store"]["degraded"]:
        print(f"  ! {meta['store']['note']}")


def _print_answer(result) -> None:
    print(f"\n问题：{result.question}")
    print("召回结果：")
    for hit in result.hits:
        snippet = hit.text.replace("\n", " ")[:90]
        print(f"  [{hit.rank}] {hit.chunk_id} score={hit.score:.3f} {hit.doc}/{hit.heading} :: {snippet}")
    print(f"\n回答（{result.generation.mode}）：\n{result.generation.answer}")
    if result.generation.refused:
        print("\n引用校验：系统按「资料不足」拒答，未生成可核验断言")
    else:
        print(
            f"\n引用校验：{result.verification.supported}/{result.verification.total} 条断言已支撑，"
            f"支撑率 {result.verification.support_rate:.1%}"
        )
    if result.verification.unknown_ids:
        print(f"  越界引用：{result.verification.unknown_ids}")
    for note in result.notes:
        print(f"  提示：{note}")


def cmd_query(cfg: Config, question: str, k: int, require_citation: bool) -> None:
    pipeline = RagPipeline(cfg)
    _print_answer(pipeline.ask(question, k=k, require_citation=require_citation))


DEMO_QUESTIONS = [
    "快速排序的最坏时间复杂度是多少，什么情况下会退化？",
    "pandas 里 merge 的 how 参数支持哪几种连接方式？",
    "BGE 中文小模型的输出向量维度是多少，查询侧应该怎么处理？",
    "公司差旅报销的住宿标准是多少钱一晚？",
]


def cmd_demo(cfg: Config) -> None:
    if not cfg.index_meta_path.exists():
        print("[demo] 未发现索引，先自动构建……")
        cmd_build(cfg)
    pipeline = RagPipeline(cfg)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    results = [pipeline.ask(q, k=cfg.top_k) for q in DEMO_QUESTIONS]

    md = [
        "# 最小 RAG 演示结果",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 索引配置：chunk_size={cfg.chunk_size}，overlap={cfg.overlap}，top_k={cfg.top_k}",
        f"- 向量模型：{pipeline.meta['embedder']['name']}（降级={pipeline.meta['embedder']['degraded']}）",
        f"- 向量库：{pipeline.meta['store']['kind']}（降级={pipeline.meta['store']['degraded']}）",
        f"- 生成模式：{results[0].generation.mode}",
        "",
    ]
    for idx, result in enumerate(results, start=1):
        md.append(f"## 示例 {idx}：{result.question}")
        md.append("")
        md.append("| 排名 | chunk | 相似度 | 来源 |")
        md.append("| --- | --- | --- | --- |")
        for hit in result.hits:
            md.append(
                f"| {hit.rank} | {hit.chunk_id} | {hit.score:.3f} | {hit.doc} · {hit.heading} |"
            )
        md.append("")
        md.append("**回答**")
        md.append("")
        md.append(result.generation.answer)
        md.append("")
        v = result.verification
        if result.generation.refused:
            md.append("**引用校验**：未生成断言，系统按「资料不足」拒答")
            md.append("")
            md.append("提示：" + "；".join(result.notes))
            md.append("")
            continue
        md.append(
            f"**引用校验**：{v.supported}/{v.total} 条断言已支撑，支撑率 {v.support_rate:.1%}"
            + (f"；越界引用 {v.unknown_ids}" if v.unknown_ids else "")
        )
        md.append("")
        md.append("| 断言 | 引用块 | 覆盖率 | 结论 |")
        md.append("| --- | --- | --- | --- |")
        for claim in v.claims:
            md.append(
                f"| {claim.claim} | {','.join(claim.cited_ids) or '-'} | "
                f"{claim.best_score:.2f} | {'已支撑' if claim.supported else '未支撑'} |"
            )
        md.append("")
        if result.notes:
            md.append("提示：" + "；".join(result.notes))
            md.append("")

    (cfg.outputs_dir / "demo.md").write_text("\n".join(md), encoding="utf-8")
    (cfg.outputs_dir / "demo.json").write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[demo] 已写出 outputs/demo.md 与 outputs/demo.json（{len(results)} 个示例）")


def cmd_eval(cfg: Config, limit: int) -> None:
    from rag.chunker import load_paragraphs
    from rag.evaluate import evaluate, load_qa, resolve_gold_ids, save_eval_outputs

    if not cfg.index_meta_path.exists():
        print("[eval] 未发现索引，先自动构建……")
        cmd_build(cfg)
    pipeline = RagPipeline(cfg)
    qa = resolve_gold_ids(load_qa(cfg), load_paragraphs(cfg.paragraphs_path))
    print(f"[eval] 评测题数：{len(qa)}")

    baseline = evaluate(
        pipeline, cfg, qa, k=1, require_citation=False, label="基线（Top-1 无引用约束）", limit=limit
    )
    full = evaluate(
        pipeline,
        cfg,
        qa,
        k=cfg.top_k,
        require_citation=True,
        label=f"完整流程（Top-{cfg.top_k} 强制引用+校验）",
        limit=limit,
    )
    path = save_eval_outputs([baseline, full], cfg, pipeline.meta)

    for r in (baseline, full):
        print(
            f"  {r['label']}：命中率@{r['k']}={r['hit_rate_at_k']:.2%}，"
            f"答案正确率={r['answer_accuracy']:.2%}，引用命中率={r['citation_accuracy']:.2%}，"
            f"支撑率={r['support_rate']:.2%}"
        )
    print(f"[eval] 报告已写入 {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小中文 RAG 项目")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="构建切块与向量索引")

    q = sub.add_parser("query", help="检索并回答单个问题")
    q.add_argument("question")
    q.add_argument("--no-citation", action="store_true", help="关闭引用约束")

    sub.add_parser("demo", help="一键跑通示例并写入 outputs/")
    e = sub.add_parser("eval", help="运行 30 题评测并生成报告")
    e.add_argument("--limit", type=int, default=0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.overlap = args.overlap
    if args.k:
        cfg.top_k = args.k

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
