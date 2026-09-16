"""起動済みCompose環境を公開API経由で検証し、内容を除いた証拠を保存する。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:18080"
REPORT = ROOT / "reports" / "compose-smoke.json"
API_KEY = "local-compose-demo-key"


def _request(path: str, *, payload: dict[str, str] | None = None) -> tuple[int, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(  # noqa: S310 - 固定されたloopback URLのみ。
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback only
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> int:
    health_status, health = _request("/healthz")
    triage_status, triage = _request(
        "/v1/triage",
        payload={
            "external_id": "compose-smoke-001",
            "subject": "Refund request",
            "body": "Please refund order ORD-10042. I was charged $42.50.",
        },
    )
    run_id = cast(dict[str, Any], triage).get("run_id")
    trace_status, trace = _request(f"/v1/runs/{run_id}/trace")
    trace_body = cast(dict[str, Any], trace)
    report = {
        "summary": {"passed": 3, "total": 3},
        "checks": {
            "health": {
                "http_status": health_status,
                "status": health.get("status"),
                "provider": health.get("provider"),
            },
            "triage": {
                "http_status": triage_status,
                "status": triage.get("status"),
                "category": triage.get("category"),
                "provider_calls": triage.get("provider_call_count"),
            },
            "trace": {
                "http_status": trace_status,
                "step_kinds": [step["kind"] for step in trace_body.get("steps", [])],
                "evidence_count": len(trace_body.get("evidence", [])),
                "tool_count": len(trace_body.get("tools", [])),
            },
        },
    }
    passed = (
        health_status == 200
        and health.get("status") == "ok"
        and triage_status == 200
        and triage.get("status") in {"auto_resolved", "needs_human_review"}
        and trace_status == 200
        and bool(trace_body.get("steps"))
        and bool(trace_body.get("evidence"))
    )
    if not passed:
        report["summary"] = {"passed": 0, "total": 3}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
