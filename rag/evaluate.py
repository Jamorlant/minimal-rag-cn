"""评测：命中率@K、答案正确率、引用命中率、断言支撑率，以及关引用约束的基线对比。"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

from .chunker import load_paragraphs
from .config import Config
from .pipeline import RagPipeline
from .textutil import normalize


def load_qa(cfg: Config) -> List[dict]:
    rows = []
    with cfg.qa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_gold_ids(qa_rows: Sequence[dict], paragraphs: Sequence) -> List[dict]:
    """把标准答案的来源片段解析为段落 ID，并在评测前校验可解析性。"""
    resolved = []
    for row in qa_rows:
        snippet = row["gold_snippet"]
        matches = [p.id for p in paragraphs if snippet in p.text]
        if not matches:
            raise ValueError(f"评测题 {row['id']} 的 gold_snippet 未在语料中找到：{snippet}")
        item = dict(row)
        item["gold_ids"] = matches
        resolved.append(item)
    return resolved


def keyword_hit(answer: str, keywords: Sequence[str]) -> bool:
    compact = normalize(answer)
    return all(normalize(kw) in compact for kw in keywords) if keywords else False


def evaluate(
    pipeline: RagPipeline,
    cfg: Config,
    qa_rows: Sequence[dict],
    k: int,
    require_citation: bool,
    label: str,
    limit: int = 0,
) -> Dict:
    rows = qa_rows[:limit] if limit else qa_rows
    details = []
    t0 = time.perf_counter()
    for item in rows:
        result = pipeline.ask(item["question"], k=k, require_citation=require_citation)
        retrieved_gold = [
            hit for hit in result.hits if set(hit.para_ids) & set(item["gold_ids"])
        ]
        first_gold_rank = None
        for hit in result.hits:
            if set(hit.para_ids) & set(item["gold_ids"]):
                first_gold_rank = hit.rank
                break
        cited_gold = [
            cid
            for cid in result.generation.citations
            if set(next((h.para_ids for h in result.hits if h.chunk_id == cid), []))
            & set(item["gold_ids"])
        ]
        details.append(
            {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": item["question"],
                "expected": item["answer"],
                "answer": result.generation.answer,
                "generation_mode": result.generation.mode,
                "hit_at_k": bool(retrieved_gold),
                "first_gold_rank": first_gold_rank,
                "answer_correct": keyword_hit(result.generation.answer, item.get("answer_keywords", [])),
                "citation_ids": result.generation.citations,
                "citation_correct": bool(cited_gold),
                "refused": result.generation.refused,
                "support_rate": 0.0 if result.generation.refused else result.verification.support_rate,
                "unsupported_claims": result.verification.total - result.verification.supported,
                "unknown_citation_ids": result.verification.unknown_ids,
                "top_chunk": result.hits[0].chunk_id if result.hits else "",
                "top_score": round(result.hits[0].score, 4) if result.hits else 0.0,
            }
        )
    elapsed = time.perf_counter() - t0
    n = len(details) or 1
    summary = {
        "label": label,
        "k": k,
        "require_citation": require_citation,
        "num_questions": len(details),
        "hit_rate_at_k": round(sum(d["hit_at_k"] for d in details) / n, 4),
        "mrr": round(
            sum((1.0 / d["first_gold_rank"]) if d["first_gold_rank"] else 0.0 for d in details) / n, 4
        ),
        "answer_accuracy": round(sum(d["answer_correct"] for d in details) / n, 4),
        "citation_accuracy": round(sum(d["citation_correct"] for d in details) / n, 4),
        "support_rate": round(statistics.mean([d["support_rate"] for d in details]), 4),
        "answers_with_citation": sum(1 for d in details if d["citation_ids"]),
        "refusals": sum(1 for d in details if d["refused"]),
        "avg_latency_ms": round(elapsed / n * 1000, 1),
        "details": details,
    }
    return summary


def build_markdown_report(results: Sequence[Dict], cfg: Config, meta: dict) -> str:
    lines = ["# 评测报告", ""]
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- 切块：chunk_size={cfg.chunk_size}，overlap={cfg.overlap}")
    lines.append(f"- 向量模型：{meta['embedder']['name']}（降级={meta['embedder']['degraded']}）")
    lines.append(f"- 向量库：{meta['store']['kind']}（降级={meta['store']['degraded']}）")
    lines.append(f"- 断言支撑阈值：{cfg.support_threshold}")
    lines.append("")
    lines.append("## 结果对比")
    lines.append("")
    lines.append("| 配置 | Top-K | 命中率@K | MRR | 答案正确率 | 引用命中率 | 断言支撑率 | 平均耗时(ms/题) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['k']} | {r['hit_rate_at_k']:.2%} | {r['mrr']:.3f} | "
            f"{r['answer_accuracy']:.2%} | {r['citation_accuracy']:.2%} | "
            f"{r['support_rate']:.2%} | {r['avg_latency_ms']} |"
        )
    lines.append("")
    lines.append(
        "> 引用命中率 = 答案中至少一个被引用块的来源段落命中标准来源的题目占比；"
        "基线不输出引用，故该项为 0，支撑校验也无从判定。"
    )
    lines.append("")
    for r in results:
        lines.append(f"## 明细：{r['label']}")
        lines.append("")
        lines.append("| 题号 | 类别 | 命中 | 首个命中位次 | 答案正确 | 引用命中 | 支撑率 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for d in r["details"]:
            lines.append(
                f"| {d['id']} | {d['category']} | {'是' if d['hit_at_k'] else '否'} | "
                f"{d['first_gold_rank'] or '-'} | {'是' if d['answer_correct'] else '否'} | "
                f"{'是' if d['citation_correct'] else '否'} | {d['support_rate']:.2%} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_eval_outputs(results: Sequence[Dict], cfg: Config, meta: dict) -> Path:
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.outputs_dir / "eval_report.md"
    report_path.write_text(build_markdown_report(results, cfg, meta), encoding="utf-8")
    details_path = cfg.outputs_dir / "eval_details.json"
    details_path.write_text(
        json.dumps([{k: v for k, v in r.items()} for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path
