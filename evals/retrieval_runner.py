"""BM25 と optional hybrid を同一データで比較する検索評価ランナー。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.rag.dense import DenseIndex
from app.rag.embedding import OllamaEmbedder
from app.rag.hybrid import HybridRetriever
from app.rag.index import KnowledgeIndex
from app.rag.loader import load_documents
from app.rag.protocol import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case_id: str
    answerable: bool
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    recall_at_k: float | None
    reciprocal_rank: float | None
    empty_result_correct: bool | None
    latency_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def run_retrieval_benchmark(
    retriever: Retriever,
    cases_path: Path,
    output_path: Path,
    *,
    mode: str,
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results: list[RetrievalCaseResult] = []

    for case in cases:
        started = time.perf_counter()
        evidence = retriever.search(case["query"], top_k=top_k, min_score=min_score)
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved = tuple(item.chunk.chunk_id for item in evidence.items)
        scores = tuple(item.score for item in evidence.items)
        relevant = set(case["relevant_chunk_ids"])
        if relevant:
            matches = relevant.intersection(retrieved)
            first_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant),
                None,
            )
            recall = len(matches) / len(relevant)
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
            empty_correct = None
        else:
            recall = None
            reciprocal_rank = None
            empty_correct = not retrieved
        results.append(
            RetrievalCaseResult(
                case_id=case["id"],
                answerable=bool(relevant),
                retrieved_chunk_ids=retrieved,
                retrieved_scores=scores,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                empty_result_correct=empty_correct,
                latency_ms=round(latency_ms, 3),
            )
        )

    answerable = [item for item in results if item.answerable]
    unsupported = [item for item in results if not item.answerable]
    latencies = [item.latency_ms for item in results]
    report: dict[str, Any] = {
        "configuration": {"mode": mode, "top_k": top_k, "min_score": min_score},
        "summary": {
            "cases": len(results),
            "answerable_cases": len(answerable),
            "unsupported_cases": len(unsupported),
            "recall_at_k": round(
                statistics.fmean(item.recall_at_k or 0.0 for item in answerable), 4
            ),
            "mrr": round(statistics.fmean(item.reciprocal_rank or 0.0 for item in answerable), 4),
            "empty_result_accuracy": round(
                statistics.fmean(float(bool(item.empty_result_correct)) for item in unsupported),
                4,
            ),
            "latency_ms_p50": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
        },
        "results": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--cases", type=Path, default=Path("evals/retrieval_cases.json"))
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge"))
    parser.add_argument("--output", type=Path, default=Path("evals/reports/retrieval-latest.json"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--semantic-min-score", type=float, default=0.55)
    parser.add_argument("--semantic-gate", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()

    documents = load_documents(args.knowledge)
    lexical = KnowledgeIndex(documents)
    retriever: Retriever = lexical
    if args.mode in {"dense", "hybrid"}:
        embedder = OllamaEmbedder(
            model=args.embedding_model or "embeddinggemma", base_url=args.ollama_url
        )
        dense = DenseIndex(documents, embedder)
        if args.mode == "dense":
            retriever = dense
        else:
            retriever = HybridRetriever(
                lexical,
                dense,
                lexical_min_score=args.min_score,
                semantic_min_score=args.semantic_min_score,
                semantic_gate=args.semantic_gate,
            )

    report = run_retrieval_benchmark(
        retriever,
        args.cases,
        args.output,
        mode=(
            f"{args.mode}:{embedder.model_id}:semantic-gate={args.semantic_gate}"
            if args.mode in {"dense", "hybrid"}
            else args.mode
        ),
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
