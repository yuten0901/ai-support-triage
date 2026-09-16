from pathlib import Path

from scripts.run_failure_demo import run_demo


def test_failure_demo_proves_all_documented_boundaries(tmp_path: Path) -> None:
    report = run_demo(tmp_path)

    assert report["summary"] == {"passed": 5, "total": 5}
    assert all(scenario["passed"] for scenario in report["scenarios"])
