---
project: ai-support-triage
type: portfolio
status: prod
gate: none
updated: 2026-08-24
next: STATUS.md 本文を参照
---

# Status

- State: production-ready portfolio demonstration; market-aligned v2 slices 1-2 implemented
- Conclusion date: 2026-09-16
- Result: tenant data/retrieval boundaries, a replaceable retrieval contract, dense/RRF experiment
  path, a measured BM25 baseline, and content-free tenant metrics are implemented; 43 tests pass;
  offline evaluation 8/8;
  seeded defects detected 4/4.
- Reusable assets: bounded structured-calling client, grounded citation validator, typed tool
  registry, deterministic policy gate, trace persistence, and offline evaluation runner.
- Lesson: outcome taxonomy and validation boundaries must be explicit before adding an LLM; a
  fake provider is useful only when it exercises the same contracts and workflow as production.
- Honest gaps: real Anthropic calls and local PostgreSQL were unavailable; Anthropic is implemented
  and type-checked, while PostgreSQL is assigned to CI.
- Next: run the optional hybrid benchmark against local Ollama, then add the pgvector persistence
  mode only if the measured quality justifies it. Multi-tenant support is implemented and tested
  locally; do not advertise hybrid superiority until its comparative acceptance criteria pass.
