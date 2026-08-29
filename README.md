# AI Support Triage

[![CI](https://github.com/yuten0901/ai-support-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/yuten0901/ai-support-triage/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

**A support team stops answering the same policy question by hand — and every automated answer cites the exact policy section it came from, or refuses to answer.**

An evidence-grounded customer-support triage API. It classifies a ticket, retrieves versioned
policy sections, validates read-only tool calls, verifies citations, and applies deterministic
approval rules before any write action. The default provider is deterministic and offline; the
same boundary supports Anthropic when credentials are supplied.

## What this demonstrates

- Six explicit outcomes, including separate system failure, model rejection, insufficient
  evidence, and valid no-action states.
- A bounded state machine with independent transport retry, output repair, call-count, deadline,
  and cost budgets.
- Strict structured outputs plus semantic citation validation against the exact retrieved chunks.
- Read-only model tools with per-tool argument schemas; write actions remain behind policy gates.
- SQLite for a zero-service local demo and PostgreSQL in CI.
- Persisted steps, provider calls, evidence usage, tool results, token usage, cost, and review state.
- A deterministic evaluation set and four seeded-defect checks.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
.\.venv\Scripts\uvicorn.exe app.api.main:app --port 8000
```

The default key is only for the local demo:

```powershell
$headers = @{ "X-API-Key" = "dev-triage-api-key" }
$body = @{
  external_id = "demo-001"
  subject = "Refund request"
  body = "Please refund order ORD-10042. I was charged `$42.50."
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/triage -Method Post -Headers $headers `
  -ContentType application/json -Body $body
```

Open `/docs` for the interactive contract. Useful endpoints are `GET /healthz`,
`POST /v1/triage`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/trace`, `GET /v1/reviews`,
`POST /v1/reviews/{id}`, and `GET /v1/knowledge`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app tests evals scripts
.\.venv\Scripts\python.exe -m pytest -q --cov=app
.\.venv\Scripts\python.exe -m evals.runner
.\.venv\Scripts\python.exe scripts\verify_mutations.py
```

The suite currently contains 27 passing tests, and the checked-in evaluation report records 8/8
passing cases. Mutation verification proves that
tests detect seeded defects in strict output validation, citation grounding, tool argument
validation, and the retry boundary.

## Real provider and deployment notes

Set `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and a supported `LLM_MODEL` to use Anthropic.
This path is implemented and type-checked but was not called locally because no paid credential
was available. Set `DATABASE_URL=postgresql+psycopg://...` for PostgreSQL; CI exercises it with a
service container. Never expose the development API key or commit `.env`.

## Scope

The ledger is an in-process stand-in for payment and ticketing APIs. It demonstrates policy
gating and idempotency boundaries, not durable payment execution. The BM25 corpus is intentionally
small and reviewable; this project does not claim a vector store is needed for five policy files.

See [architecture](docs/architecture.md) and [evaluation](docs/evaluation.md).

## License

MIT. See `LICENSE`.
