"""購入者向けに、主要な障害境界を一度に再現するデモ。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient

from app.ai.provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    TokenUsage,
)
from app.api.main import app
from app.config import reset_settings_cache
from app.services import Services

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "failure-demo.json"


class AlwaysUnavailable:
    """再試行可能なプロバイダー停止を決定的に再現する。"""

    name = "unavailable-demo"

    def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise ProviderUnavailable("simulated provider outage", request_id="demo-outage")


class AlwaysMalformed:
    """修復回数が有限であることを示すため、不正JSONだけを返す。"""

    name = "malformed-demo"

    def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        return LLMResponse(
            text="this is not json",
            usage=TokenUsage(input_tokens=12, output_tokens=4),
            model="malformed-demo-v1",
            request_id="demo-malformed",
        )


@contextmanager
def temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    """デモで変更した環境変数を、成功・失敗にかかわらず復元する。"""

    original = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    reset_settings_cache()
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings_cache()


def _set_provider(provider: LLMProvider) -> None:
    services = cast(Services, app.state.services)
    for orchestrator in services.orchestrators.values():
        orchestrator._provider = provider  # デモ専用の故障注入点。


def _ticket(external_id: str, body: str) -> dict[str, str]:
    return {"external_id": external_id, "subject": "Failure demo", "body": body}


def _scenario(name: str, passed: bool, observed: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "observed": dict(observed)}


def run_demo(workdir: Path) -> dict[str, Any]:
    """隔離された一時DBと2テナントで、5つのシナリオを実行する。"""

    knowledge_a = workdir / "knowledge-a"
    knowledge_b = workdir / "knowledge-b"
    data_a = workdir / "data-a"
    data_b = workdir / "data-b"
    shutil.copytree(ROOT / "knowledge", knowledge_a)
    shutil.copytree(ROOT / "knowledge", knowledge_b)
    shutil.copytree(ROOT / "data", data_a)
    shutil.copytree(ROOT / "data", data_b)

    environment = {
        "DATABASE_URL": f"sqlite+pysqlite:///{workdir / 'failure-demo.sqlite3'}",
        "TENANT_API_KEYS": json.dumps(
            {"tenant-a": "failure-demo-a", "tenant-b": "failure-demo-b"}
        ),
        "TENANT_KNOWLEDGE_DIRS": json.dumps(
            {"tenant-a": str(knowledge_a), "tenant-b": str(knowledge_b)}
        ),
        "TENANT_DATA_DIRS": json.dumps(
            {"tenant-a": str(data_a), "tenant-b": str(data_b)}
        ),
        "LLM_MAX_TRANSPORT_ATTEMPTS": "2",
        "LLM_MAX_REPAIR_ATTEMPTS": "1",
        "LLM_MAX_PROVIDER_CALLS_PER_STEP": "3",
        "LLM_RETRY_BASE_DELAY_SECONDS": "0.001",
        "LLM_RETRY_MAX_DELAY_SECONDS": "0.001",
    }
    auth_a = {"X-API-Key": "failure-demo-a"}
    auth_b = {"X-API-Key": "failure-demo-b"}

    with temporary_environment(environment), TestClient(app) as client:
        scenarios: list[dict[str, Any]] = []

        unsupported = client.post(
            "/v1/triage",
            headers=auth_a,
            json=_ticket("unsupported", "How do I calibrate a lunar radio telescope?"),
        ).json()
        scenarios.append(
            _scenario(
                "unsupported query refuses to invent evidence",
                unsupported["status"] == "insufficient_evidence",
                {
                    "status": unsupported["status"],
                    "provider_calls": unsupported["provider_call_count"],
                },
            )
        )

        duplicate_body = _ticket("duplicate", "Thank you, everything is resolved.")
        duplicate_first = client.post("/v1/triage", headers=auth_a, json=duplicate_body).json()
        duplicate_second = client.post("/v1/triage", headers=auth_a, json=duplicate_body).json()
        scenarios.append(
            _scenario(
                "duplicate delivery reuses the original run",
                duplicate_first["run_id"] == duplicate_second["run_id"],
                {
                    "same_run_id": duplicate_first["run_id"] == duplicate_second["run_id"],
                    "status": duplicate_first["status"],
                },
            )
        )

        cross_tenant_status = client.get(
            f"/v1/runs/{duplicate_first['run_id']}", headers=auth_b
        ).status_code
        scenarios.append(
            _scenario(
                "cross-tenant run access is indistinguishable from missing",
                cross_tenant_status == 404,
                {"http_status": cross_tenant_status},
            )
        )

        _set_provider(AlwaysUnavailable())
        outage = client.post(
            "/v1/triage",
            headers=auth_a,
            json=_ticket("outage", "Where is order ORD-10042?"),
        ).json()
        scenarios.append(
            _scenario(
                "provider outage stops after the retry budget",
                outage["status"] == "failed"
                and outage["error_kind"] == "provider_unavailable"
                and outage["provider_call_count"] == 2,
                {
                    "status": outage["status"],
                    "error_kind": outage["error_kind"],
                    "provider_calls": outage["provider_call_count"],
                },
            )
        )

        _set_provider(AlwaysMalformed())
        malformed = client.post(
            "/v1/triage",
            headers=auth_a,
            json=_ticket("malformed", "Where is order ORD-10042?"),
        ).json()
        scenarios.append(
            _scenario(
                "malformed model output stops after the repair budget",
                malformed["status"] == "failed"
                and malformed["error_kind"] == "invalid_model_output"
                and malformed["provider_call_count"] == 2,
                {
                    "status": malformed["status"],
                    "error_kind": malformed["error_kind"],
                    "provider_calls": malformed["provider_call_count"],
                },
            )
        )

    return {
        "summary": {
            "passed": sum(1 for scenario in scenarios if scenario["passed"]),
            "total": len(scenarios),
        },
        "scenarios": scenarios,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ai-support-triage-failure-demo-") as raw:
        report = run_demo(Path(raw))
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for scenario in report["scenarios"]:
        mark = "PASS" if scenario["passed"] else "FAIL"
        print(f"[{mark}] {scenario['name']}: {scenario['observed']}")
    print(f"\n{report['summary']['passed']}/{report['summary']['total']} scenarios passed")
    if report["summary"]["passed"] != report["summary"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
