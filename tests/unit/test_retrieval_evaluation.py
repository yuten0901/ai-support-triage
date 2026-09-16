import json
from pathlib import Path

from app.rag.index import KnowledgeIndex
from app.rag.loader import load_documents
from evals.retrieval_runner import run_retrieval_benchmark


def test_bm25_retrieval_benchmark_has_expected_security_gate(tmp_path: Path) -> None:
    report = run_retrieval_benchmark(
        KnowledgeIndex(load_documents(Path("knowledge"))),
        Path("evals/retrieval_cases.json"),
        tmp_path / "report.json",
        mode="bm25",
        top_k=4,
        min_score=0.15,
    )

    summary = report["summary"]
    assert summary["cases"] == 13
    # Checked-in baseline, not aspirational thresholds. The damaged-item paraphrase
    # and printer false positive are the explicit reasons to test a hybrid path.
    assert summary["recall_at_k"] == 0.9
    assert summary["mrr"] == 0.85
    assert summary["empty_result_accuracy"] == 0.6667
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["summary"] == summary
    assert len(written["results"]) == 13
