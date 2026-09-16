from pathlib import Path

from evals.runner import run


def test_checked_in_evaluation_suite_passes(tmp_path: Path) -> None:
    report = run(Path("evals/cases.json"), tmp_path / "report.json")

    assert report["summary"] == {"cases": 8, "passed": 8, "failed": 0, "pass_rate": 1.0}
