# Decisions

## 2026-08-24

- Use a deterministic in-process provider by default so the demo, tests, and evaluation require no
  paid key and remain reproducible. Keep Anthropic behind the same provider protocol.
- Use BM25 over versioned Markdown because the corpus is small and lexical; avoid an unjustified
  vector database.
- Treat model rejection, insufficient evidence, no action, human review, automatic resolution,
  and system failure as separate persisted outcomes.
- Keep read tools model-accessible and write actions policy-accessible only.
- Use SQLite locally and PostgreSQL in CI because neither Docker nor PostgreSQL was available in
  the local environment.

## 2026-09-16

- Keep BM25 as the honest default for the bundled small corpus. Do not replace it merely to add a
  vector-database keyword to the portfolio.
- Design an optional production-scale mode around tenant-scoped retrieval, measured hybrid search,
  leakage/injection evaluation, and operability evidence. The design and release gates are in
  `docs/market-aligned-upgrade-plan.md`.
- Apply tenant scope before retrieval and authorize tenant identity from server-side credentials;
  filtering a global top-k or trusting a request-provided tenant id is explicitly rejected.
- Treat the upgrade as a large implementation: finish design first, then implement and independently
  re-review it before updating public claims.
- Complete the tenant boundary before adding retrieval complexity. Tenant identity now derives from
  credentials and scopes persistence, idempotency, tools, knowledge, traces, and reviews.
- Depend on a typed `Retriever` contract in the workflow. Keep `KnowledgeIndex` as the verified BM25
  implementation so a later hybrid experiment cannot silently change orchestration semantics.
- Remove injection-shaped retrieved chunks before either tool planning or response generation,
  persist the blocked count in the trace, and retain the signal in the deterministic action gate.
  This reduces a known path; it is not described as complete prompt-injection prevention.
- Keep BM25 as the default but expose an optional Ollama `all-minilm` hybrid mode. On the fixed
  13-case set, semantic score 0.20 cleanly separated answerable from unsupported queries; using it
  as an out-of-domain gate improved Recall@4 from 0.90 to 1.00, MRR from 0.85 to 0.95, and empty
  accuracy from 0.6667 to 1.00, at roughly 62 ms p50 instead of sub-millisecond BM25. This small
  dataset is regression evidence, not a production-quality estimate.
- Make operational claims reproducible through an API-level script rather than screenshots alone.
  The script uses an isolated temporary database, injects deterministic provider failures, and
  writes a checked-in 5/5 report covering unsupported, duplicate, cross-tenant, outage, and invalid
  model-output behavior. This report will also be the source for the captioned walkthrough.
