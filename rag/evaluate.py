"""评测：命中率@K、MRR、答案正确率、引用命中率、断言支撑率，以及关闭引用约束的基线对比。

所有指标在同一个 pipeline 上跑，基线与完整流程只差 `k` 与 `require_citation`。
"""
from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .chunker import Paragraph, load_paragraphs
from .config import Config
from .pipeline import RagPipeline
from .store import RetrievedChunk
from .textutil import normalize


def load_qa(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.qa_path.exists():
        raise FileNotFoundError(f"评测题文件不存在：{cfg.qa_path}")
    rows: list[dict[str, Any]] = []
    with cfg.qa_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_gold_ids(
    qa_rows: Sequence[dict[str, Any]], paragraphs: Sequence[Paragraph]
) -> list[dict[str, Any]]:
    """把标准答案片段解析为段落 ID；解析失败直接报错，避免语料漂移后标准答案失效。"""
    resolved: list[dict[str, Any]] = []
    for row in qa_rows:
        snippet = row["gold_snippet"]
        matches = [paragraph.id for paragraph in paragraphs if snippet in paragraph.text]
        if not matches:
            raise ValueError(f"评测题 {row['id']} 的 gold_snippet 未在语料中找到：{snippet}")
        resolved.append({**row, "gold_ids": matches})
    return resolved


def keyword_hit(answer: str, keywords: Sequence[str]) -> bool:
    compact = normalize(answer)
    return all(normalize(keyword) in compact for keyword in keywords) if keywords else False


def _first_gold_rank(hits: Sequence[RetrievedChunk], gold_ids: Sequence[str]) -> int | None:
    for hit in hits:
        if set(hit.para_ids) & set(gold_ids):
            return hit.rank
    return None


def _cited_gold(
    citations: Sequence[str], hits: Sequence[RetrievedChunk], gold_ids: Sequence[str]
) -> bool:
    for chunk_id in citations:
        para_ids = next((hit.para_ids for hit in hits if hit.chunk_id == chunk_id), [])
        if set(para_ids) & set(gold_ids):
            return True
    return False


def _score_question(
    pipeline: RagPipeline,
    item: dict[str, Any],
    k: int,
    require_citation: bool,
) -> dict[str, Any]:
    result = pipeline.ask(item["question"], k=k, require_citation=require_citation)
    gold_ids = item["gold_ids"]
    first_gold_rank = _first_gold_rank(result.hits, gold_ids)
    return {
        "id": item["id"],
        "category": item.get("category", ""),
        "question": item["question"],
        "expected": item["answer"],
        "answer": result.generation.answer,
        "generation_mode": result.generation.mode,
        "hit_at_k": first_gold_rank is not None,
        "first_gold_rank": first_gold_rank,
        "answer_correct": keyword_hit(result.generation.answer, item.get("answer_keywords", [])),
        "citation_ids": result.generation.citations,
        "citation_correct": _cited_gold(result.generation.citations, result.hits, gold_ids),
        "refused": result.generation.refused,
        # 拒答视为未支撑，避免「不回答也能拿高支撑率」
        "support_rate": 0.0 if result.generation.refused else result.verification.support_rate,
        "unsupported_claims": result.verification.total - result.verification.supported,
        "unknown_citation_ids": result.verification.unknown_ids,
        "top_chunk": result.hits[0].chunk_id if result.hits else "",
        "top_score": round(result.hits[0].score, 4) if result.hits else 0.0,
    }


def _summarize(
    details: Sequence[dict[str, Any]],
    label: str,
    k: int,
    require_citation: bool,
    elapsed: float,
) -> dict[str, Any]:
    count = len(details) or 1
    return {
        "label": label,
        "k": k,
        "require_citation": require_citation,
        "num_questions": len(details),
        "hit_rate_at_k": round(sum(1 for row in details if row["hit_at_k"]) / count, 4),
        "mrr": round(
            sum(1.0 / row["first_gold_rank"] if row["first_gold_rank"] else 0.0 for row in details)
            / count,
            4,
        ),
        "answer_accuracy": round(sum(1 for row in details if row["answer_correct"]) / count, 4),
        "citation_accuracy": round(sum(1 for row in details if row["citation_correct"]) / count, 4),
        "support_rate": round(statistics.mean([row["support_rate"] for row in details]) if details else 0.0, 4),
        "answers_with_citation": sum(1 for row in details if row["citation_ids"]),
        "refusals": sum(1 for row in details if row["refused"]),
        "avg_latency_ms": round(elapsed / count * 1000, 1),
        "details": list(details),
    }


def evaluate(
    pipeline: RagPipeline,
    cfg: Config,
    qa_rows: Sequence[dict[str, Any]],
    k: int,
    require_citation: bool,
    label: str,
    limit: int = 0,
) -> dict[str, Any]:
    rows = list(qa_rows[:limit]) if limit else list(qa_rows)
    started = time.perf_counter()
    details = [_score_question(pipeline, item, k, require_citation) for item in rows]
    elapsed = time.perf_counter() - started
    return _summarize(details, label, k, require_citation, elapsed)


def build_markdown_report(
    results: Sequence[dict[str, Any]], cfg: Config, meta: dict[str, Any]
) -> str:
    lines = ["# 评测报告", "", "## 运行配置", ""]
    lines.append(f"- 切块：chunk_size={cfg.chunk_size}，overlap={cfg.overlap}")
    lines.append(f"- 向量模型：{meta['embedder']['name']}（降级={meta['embedder']['degraded']}）")
    lines.append(f"- 向量库：{meta['store']['kind']}（降级={meta['store']['degraded']}）")
    lines.append(f"- 断言支撑阈值：{cfg.support_threshold}")
    lines.append("")
    lines.append("## 结果对比")
    lines.append("")
    lines.append("| 配置 | Top-K | 命中率@K | MRR | 答案正确率 | 引用命中率 | 断言支撑率 | 平均耗时(ms/题) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        lines.append(
            f"| {result['label']} | {result['k']} | {result['hit_rate_at_k']:.2%} | "
            f"{result['mrr']:.3f} | {result['answer_accuracy']:.2%} | "
            f"{result['citation_accuracy']:.2%} | {result['support_rate']:.2%} | "
            f"{result['avg_latency_ms']} |"
        )
    lines.append("")
    lines.append(
        "> 引用命中率 = 答案中至少一个被引用块的来源段落命中标准来源的题目占比；"
        "基线不输出引用，故该项为 0，支撑校验也无从判定。"
    )
    lines.append("")
    for result in results:
        lines.append(f"## 明细：{result['label']}")
        lines.append("")
        lines.append("| 题号 | 类别 | 命中 | 首个命中位次 | 答案正确 | 引用命中 | 支撑率 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for detail in result["details"]:
            lines.append(
                f"| {detail['id']} | {detail['category']} | "
                f"{'是' if detail['hit_at_k'] else '否'} | {detail['first_gold_rank'] or '-'} | "
                f"{'是' if detail['answer_correct'] else '否'} | "
                f"{'是' if detail['citation_correct'] else '否'} | {detail['support_rate']:.2%} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_eval_outputs(
    results: Sequence[dict[str, Any]], cfg: Config, meta: dict[str, Any]
) -> Path:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.outputs_dir / "eval_report.md"
    report_path.write_text(build_markdown_report(results, cfg, meta), encoding="utf-8")
    (cfg.outputs_dir / "eval_details.json").write_text(
        json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path


def run_evaluation(pipeline: RagPipeline, cfg: Config, limit: int = 0) -> list[dict[str, Any]]:
    """基线与完整配置的固定组合，CLI 与脚本共用。"""
    qa_rows = resolve_gold_ids(load_qa(cfg), load_paragraphs(cfg.paragraphs_path))
    baseline = evaluate(
        pipeline,
        cfg,
        qa_rows,
        k=1,
        require_citation=False,
        label="基线（Top-1 无引用约束）",
        limit=limit,
    )
    full = evaluate(
        pipeline,
        cfg,
        qa_rows,
        k=cfg.top_k,
        require_citation=True,
        label=f"完整流程（Top-{cfg.top_k} 强制引用+校验）",
        limit=limit,
    )
    return [baseline, full]
