"""Deterministic offline evaluation runner."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import Settings
from app.db.session import create_all, create_db_engine, create_session_factory
from app.services import build_services
from app.workflow.grounding import validate_resolution
from app.workflow.orchestrator import TriageRequest


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    passed: bool
    status: str
    category: str | None
    grounded: bool
    provider_calls: int
    cost_usd: float | None
    failures: tuple[str, ...]


def run(cases_path: Path, output_path: Path) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory() as temp:
        settings = Settings(database_url=f"sqlite+pysqlite:///{Path(temp) / 'eval.sqlite3'}")
        engine = create_db_engine(settings)
        create_all(engine)
        services = build_services(settings, create_session_factory(engine))
        for case in cases:
            result = services.orchestrator.run(
                TriageRequest(external_id=case["id"], subject=case["subject"], body=case["body"])
            )
            failures: list[str] = []
            category = result.classification.category.value if result.classification else None
            if case.get("status") and result.status.value != case["status"]:
                failures.append(f"status: expected {case['status']}, got {result.status.value}")
            if case.get("category") and category != case["category"]:
                failures.append(f"category: expected {case['category']}, got {category}")
            grounding_error = (
                validate_resolution(result.resolution, result.evidence)
                if result.resolution
                else None
            )
            results.append(
                CaseResult(
                    case_id=case["id"],
                    passed=not failures and grounding_error is None,
                    status=result.status.value,
                    category=category,
                    grounded=grounding_error is None,
                    provider_calls=result.provider_call_count,
                    cost_usd=(
                        round(result.estimated_cost_usd, 6)
                        if result.estimated_cost_usd is not None
                        else None
                    ),
                    failures=tuple(failures + ([grounding_error] if grounding_error else [])),
                )
            )
        engine.dispose()
    passed = sum(item.passed for item in results)
    report: dict[str, object] = {
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4),
        },
        "results": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/eval-report.json"))
    args = parser.parse_args()
    report = run(args.cases, args.output)
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(0 if report["summary"]["failed"] == 0 else 1)  # type: ignore[index]


if __name__ == "__main__":
    main()
