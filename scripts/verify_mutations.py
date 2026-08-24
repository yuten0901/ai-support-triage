"""Prove that four critical test guards fail under seeded defects."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = (
    (
        "structured-validation",
        "app/ai/schemas.py",
        'ConfigDict(extra="forbid")',
        'ConfigDict(extra="ignore")',
        "tests/unit/test_core.py::test_structured_schema_rejects_extra_fields",
    ),
    (
        "citation-grounding",
        "app/workflow/grounding.py",
        "if chunk is None:",
        "if False and chunk is None:",
        "tests/unit/test_core.py::test_grounding_rejects_unknown_and_fabricated_citations",
    ),
    (
        "tool-argument-validation",
        "app/tools/registry.py",
        "args = tool.args_model.model_validate(parsed)",
        "args = tool.args_model.model_construct()",
        "tests/unit/test_core.py::test_tool_registry_rejects_unknown_tool_and_bad_json",
    ),
    (
        "retry-boundary",
        "app/ai/client.py",
        "if transport_attempts >= self._limits.max_transport_attempts:",
        "if transport_attempts > self._limits.max_transport_attempts:",
        "tests/unit/test_reliability.py::test_transport_retry_boundary_is_exact",
    ),
)


def main() -> None:
    detected = 0
    with tempfile.TemporaryDirectory() as directory:
        sandbox = Path(directory) / "repo"
        shutil.copytree(ROOT, sandbox, ignore=shutil.ignore_patterns(".git", ".venv", "*.sqlite3"))
        for name, relative, needle, replacement, test in MUTATIONS:
            path = sandbox / relative
            source = path.read_text(encoding="utf-8")
            if source.count(needle) != 1:
                raise RuntimeError(f"Mutation anchor changed: {name}")
            path.write_text(source.replace(needle, replacement), encoding="utf-8")
            process = subprocess.run(  # noqa: S603 - command components are constants above
                [sys.executable, "-m", "pytest", "-q", test],
                cwd=sandbox,
                capture_output=True,
                text=True,
                check=False,
            )
            caught = process.returncode != 0
            detected += int(caught)
            print(f"{name}: {'DETECTED' if caught else 'SURVIVED'}")
            path.write_text(source, encoding="utf-8")
    print(f"mutation score: {detected}/{len(MUTATIONS)}")
    raise SystemExit(0 if detected == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
