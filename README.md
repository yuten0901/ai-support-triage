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

**Evidence:** [client-facing case study](docs/case-study.md) ·
[architecture and trust boundaries](docs/architecture.md) ·
[evaluation method and measured limits](docs/evaluation.md)

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
- Optional credential-derived tenant isolation: separate knowledge/tool stores, tenant-scoped
  idempotency, runs, traces and review queues, with cross-tenant negative API tests.

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
`POST /v1/reviews/{id}`, `GET /v1/knowledge`, and `GET /v1/metrics`.

Optional measured hybrid retrieval runs entirely on the local machine:

```powershell
ollama pull all-minilm
$env:RETRIEVAL_MODE = "hybrid"
$env:EMBEDDING_MODEL = "all-minilm"
.\.venv\Scripts\uvicorn.exe app.api.main:app --port 8000
```

If Ollama is unavailable, leave `RETRIEVAL_MODE` unset and the service uses the verified BM25
default without any embedding download or model process.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app tests evals scripts
.\.venv\Scripts\python.exe -m pytest -q --cov=app
.\.venv\Scripts\python.exe -m evals.runner
.\.venv\Scripts\python.exe -m evals.retrieval_runner
.\.venv\Scripts\python.exe scripts\verify_mutations.py
```

The suite currently contains 45 passing tests, and the checked-in evaluation report records 8/8
passing cases. Mutation verification proves that
tests detect seeded defects in strict output validation, citation grounding, tool argument
validation, and the retry boundary. The separate retrieval baseline reports Recall@4 0.90,
MRR 0.85, and unsupported-query empty-result accuracy 0.6667; these deliberately imperfect
numbers define what the optional hybrid experiment must improve without hiding false positives.
With local Ollama `all-minilm`, the gated hybrid mode measures Recall@4 **1.00**, MRR **0.95**, and
empty-result accuracy **1.00** on the same cases (p50 **61.871 ms** versus sub-millisecond BM25).
It is therefore available as an explicit quality/latency trade-off, not silently made the default.

## Real provider and deployment notes

Set `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and a supported `LLM_MODEL` to use Anthropic.
This path is implemented and type-checked but was not called locally because no paid credential
was available. Set `DATABASE_URL=postgresql+psycopg://...` for PostgreSQL; CI exercises it with a
service container. Never expose the development API key or commit `.env`.

## Scope

The ledger is an in-process stand-in for payment and ticketing APIs. It demonstrates policy
gating and idempotency boundaries, not durable payment execution. The BM25 corpus is intentionally
small and reviewable. Set `RETRIEVAL_MODE=hybrid` with local Ollama and `all-minilm` to use the
measured semantic-gated RRF path. BM25 remains the default because it is transparent, requires no
model service, and is dramatically faster for five policy files.
Injection-shaped text retrieved from a knowledge document is removed before prompt construction and
recorded in the persisted retrieval step; this is a tested gate, not a claim that pattern matching
detects every possible indirect injection.

The authenticated metrics endpoint reports per-tenant empty-retrieval, human-review, provider and
citation-failure rates plus run/stage latency percentiles. It deliberately never loads ticket text,
prompt bodies, provider payloads, or reusable tenant identifiers into its response.

See [architecture](docs/architecture.md) and [evaluation](docs/evaluation.md).

## License

MIT. See `LICENSE`.
