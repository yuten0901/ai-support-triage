"""テナント別の、内容を含まない運用メトリクス集計。"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import TriageRun
from app.domain.states import ErrorKind, StepKind, TriageStatus

_PROVIDER_ERRORS = {
    ErrorKind.PROVIDER_TIMEOUT.value,
    ErrorKind.PROVIDER_RATE_LIMITED.value,
    ErrorKind.PROVIDER_UNAVAILABLE.value,
    ErrorKind.PROVIDER_AUTH.value,
    ErrorKind.PROVIDER_BAD_REQUEST.value,
}


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _latency_summary(values: list[int]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "average": round(statistics.fmean(values), 2) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def collect_metrics(session: Session, tenant_id: str) -> dict[str, Any]:
    """Aggregate one tenant without loading its ticket text or model prompts."""
    runs = session.scalars(
        select(TriageRun)
        .where(TriageRun.tenant_id == tenant_id)
        .options(selectinload(TriageRun.steps), selectinload(TriageRun.evidence))
    ).all()
    total = len(runs)
    retrieval_runs = [
        run for run in runs if any(step.kind == StepKind.RETRIEVE.value for step in run.steps)
    ]
    retrieval_empty = sum(not run.evidence for run in retrieval_runs)
    human_reviews = sum(run.status == TriageStatus.NEEDS_HUMAN_REVIEW.value for run in runs)
    provider_failures = sum(run.error_kind in _PROVIDER_ERRORS for run in runs)
    citation_failures = sum(run.error_kind == ErrorKind.UNGROUNDED_CITATION.value for run in runs)

    stage_latencies: defaultdict[str, list[int]] = defaultdict(list)
    for run in runs:
        for step in run.steps:
            stage_latencies[step.kind].append(step.latency_ms)

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "runs": total,
        "retrieval": {
            "attempts": len(retrieval_runs),
            "empty": retrieval_empty,
            "empty_rate": rate(retrieval_empty, len(retrieval_runs)),
        },
        "human_review": {"count": human_reviews, "rate": rate(human_reviews, total)},
        "provider_failures": {
            "count": provider_failures,
            "rate": rate(provider_failures, total),
        },
        "citation_validation_failures": {
            "count": citation_failures,
            "rate": rate(citation_failures, total),
        },
        "run_latency_ms": _latency_summary([run.latency_ms for run in runs]),
        "stage_latency_ms": {
            kind: _latency_summary(values) for kind, values in sorted(stage_latencies.items())
        },
    }
