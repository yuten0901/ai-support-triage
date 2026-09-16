# Market-aligned upgrade plan

**Status:** slices 1-4 verified; slice 3 excludes pgvector; slice 5 lacks only the walkthrough
**Date:** 2026-09-16  
**Reason:** recent Upwork fixed-price AI/RAG jobs repeatedly require tenant isolation, vector or
hybrid retrieval, leakage testing, observability, Docker deployment, and client-readable evidence.

## Decision

Do not replace the current BM25 implementation merely to claim a vector database. The current
five-document corpus is small, vocabulary-aligned, and benefits from explainable lexical scores.
Instead, add a production-scale mode with explicit tenant isolation and a measured hybrid-retrieval
comparison. Keep the current offline demo as the fast, deterministic default.

The result must demonstrate a decision, not a shopping list of AI technologies:

- BM25 remains the baseline and default for the bundled corpus.
- Dense/hybrid retrieval is optional and justified only by a retrieval benchmark.
- Tenant scope is applied before candidate retrieval. Post-filtering a global top-k is forbidden.
- A zero cross-tenant leakage result is a release gate, not a README claim.

## Target buyer story

> A support platform can serve separate organisations from one deployment without allowing one
> organisation's policy text, ticket, trace, or review to appear in another organisation's result.
> Retrieval quality is measured, citations remain verifiable, and unsafe or unsupported answers
> still route to a human.

This extends the existing evidence-grounded workflow rather than turning the repository into a
generic chatbot or full-stack SaaS.

## Scope

### 1. Tenant boundary

- Map an API credential to a server-side `tenant_id`; never accept tenant identity from a trusted
  request body.
- Add `tenant_id` to tickets, runs, reviews, evidence, and knowledge-source metadata.
- Make external ticket idempotency unique per tenant, not globally.
- Require tenant predicates in every run/review/trace query.
- Partition knowledge indexes by tenant before scoring.
- Return `404`, not `403`, for another tenant's object ID to avoid existence disclosure.

### 2. Retrieval interface and hybrid experiment

- Extract a typed `Retriever` protocol returning the existing `EvidenceSet` contract.
- Preserve the current `KnowledgeIndex` as the lexical implementation.
- Add an optional PostgreSQL/pgvector-backed dense retriever and a deterministic hybrid rank
  fusion layer.
- Store document version, tenant id, audience, checksum, and chunk id next to each embedding.
- Reindex by immutable document version and atomically switch the active version.
- Do not send ticket or policy text to a paid embedding API in the default path.

### 3. Evaluation and security gates

- Add a labelled retrieval set with answerable, paraphrased, unsupported, conflicting-audience,
  and prompt-injection-shaped queries.
- Report Recall@k, MRR, empty-result accuracy, citation validity, latency, and index size.
- Add two tenants with overlapping terminology and conflicting policy values.
- Fail CI on any retrieved/cited chunk from the wrong tenant.
- Add indirect prompt-injection text inside a knowledge document and prove it cannot alter tool or
  action policy.
- Seed defects for a missing tenant predicate and a global-before-filter top-k implementation;
  the tests must catch both.

### 4. Operability evidence

- Add structured counters for retrieval empty rate, human-review rate, provider failures,
  citation-validation failures, and per-stage latency.
- Expose a credential-protected metrics summary with no ticket text, prompt body, or personal data.
- Provide Docker Compose for the API and a pgvector-enabled PostgreSQL service.
- Run a scripted failure demonstration: provider outage, malformed model output, unsupported query,
  duplicate ticket, and cross-tenant access attempt.

### 5. Buyer-facing evidence

- Add a client-facing case study with problem, architecture, failure matrix, verified claims, and
  honest limits.
- Add one architecture diagram showing the tenant boundary before retrieval.
- Add a 90–150 second captioned walkthrough; spoken English is not required.
- Link the case study, evaluation report, failure demo, and video from the first screen of README.

## Non-goals

- A chat UI, billing, subscription management, or general SaaS administration.
- Claiming production traffic or paid-client deployment.
- AWS/Azure deployment solely to add a cloud logo.
- Fine-tuning, autonomous write-capable agents, or unbounded agent loops.
- Removing the current deterministic offline mode.

## Acceptance criteria

1. Existing tests and the original eight-case evaluation remain green.
2. All API reads/writes and knowledge retrieval are tenant-scoped by construction.
3. Cross-tenant retrieval, citation, trace access, and review access are zero in the adversarial
   test suite.
4. BM25 and hybrid retrieval are compared on the same labelled dataset; the README reports the
   measured result even if hybrid loses.
5. Unsupported queries still return no evidence rather than a forced nearest vector.
6. Citation quotes resolve against the exact retrieved, versioned chunk.
7. Docker Compose starts a clean API + pgvector database and the scripted demo completes.
8. No secret, raw provider payload, ticket body, or reusable tenant identifier appears in public
   evidence.
9. The case study distinguishes implemented/tested, locally demonstrated, and not verified.

## Delivery slices

1. Tenant model and authorization boundary, including negative tests.
2. Retriever protocol and unchanged BM25 adapter.
3. Optional dense/hybrid path plus migrations and benchmark.
4. Security/evaluation gates and seeded-defect checks.
5. Compose/runtime proof, case study, diagram, and captioned walkthrough.

Each slice must be independently reviewable. Do not begin the next slice while a tenant-boundary
finding remains open.

## Implementation record

- 2026-09-16 — Slice 1: credential-derived tenant identity, tenant-scoped persistence,
  per-tenant knowledge/tool stores, cross-tenant `404` behavior, and negative isolation tests.
- 2026-09-16 — Slice 2: typed `Retriever` protocol introduced without changing the measured BM25
  behavior or the existing `EvidenceSet`/citation contract.
- 2026-09-16 — Slice 3: local Ollama `all-minilm`, in-memory cosine search, semantic gating, and
  deterministic RRF measured on the same 13-case set. Hybrid improved all three quality metrics;
  durable pgvector behavior is still pending and must not be claimed.
- 2026-09-16 — Slice 4: indirect-injection-shaped knowledge is removed before prompt construction;
  a tenant-scoped content-free metrics endpoint reports operational rates and stage latency; and a
  5/5 API-level failure demo proves unsupported, duplicate, cross-tenant, provider-outage, and
  malformed-output behavior. Tenant isolation tests fail if tenant predicates or retrieval
  partitioning are removed. A separate mutation runner still covers four foundational defects.
- 2026-09-16 — Slice 5 in progress: the case study, architecture diagram, and machine-readable
  reports are linked from the README. The captioned walkthrough remains pending.
