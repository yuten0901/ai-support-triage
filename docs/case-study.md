# Case study: auditable AI support triage for multiple customers

## The business problem

A support operation wants to automate repetitive policy questions and low-risk actions, but a
plausible answer is not enough. Operators must know which policy version supported the answer,
why a write action was or was not executed, how provider failures differ from a genuine lack of
evidence, and whether one customer's data can reach another customer's run.

The dangerous shortcut is a generic chatbot connected to tools. It can produce fluent replies, but
it makes cost, retry behavior, tenant boundaries, citations, and approval decisions difficult to
audit. This demonstration instead treats the LLM as one bounded proposer inside a deterministic
workflow.

## What was built

- A FastAPI workflow with explicit precheck, classification, retrieval, read-tool planning,
  resolution, citation validation, policy decision, and write-action stages.
- Credential-derived tenant selection before knowledge retrieval or tool access. Ticket
  idempotency, runs, evidence, traces, and human-review queues are tenant-scoped.
- Versioned Markdown knowledge with inspectable BM25 retrieval and exact-quote citation checks.
- A replaceable retrieval contract plus local Ollama embeddings, cosine retrieval, and deterministic
  rank fusion for an honest BM25-versus-hybrid experiment.
- Separate retry, repair, call-count, deadline, and cost ceilings around model calls.
- A human-review gate for high-risk writes, low confidence, conflicting evidence, tool failures,
  and suspected direct or indirect prompt injection.
- Content-free operational metrics for empty retrieval, review routing, provider/citation failures,
  and run/stage latency.

## Failure behavior

| Condition | Observable result | Unsafe fallback avoided |
|---|---|---|
| No supporting policy | `insufficient_evidence` | Fabricating from the nearest document |
| Invalid/fabricated citation | repair, then typed failure | Returning an ungrounded answer |
| Provider outage | retry within budget, then `failed` | Reporting an outage as a business decision |
| Risky write action | pending human review | Letting the model approve its own action |
| Cross-tenant run/review ID | `404` | Revealing or returning another tenant's data |
| Injection-shaped knowledge chunk | removed and recorded before prompting | Treating retrieved text as trusted instructions |
| Duplicate external ticket ID | original run returned per tenant | Paying for and executing the same request twice |

## Measured evidence

- **46 automated tests** currently pass across contracts, reliability boundaries, API behavior,
  tenant isolation, indirect-injection filtering, and retrieval evaluation.
- A scripted API-level failure demonstration passes **5/5** scenarios covering unsupported
  questions, idempotent duplicates, cross-tenant access, provider outage, and malformed output.
- The deterministic end-to-end evaluation passes **8/8** checked-in scenarios.
- Four deliberately seeded defects are detected: weakened structured validation, fabricated
  citations, bypassed tool argument validation, and an extra retry beyond the budget.
- The BM25 retrieval baseline on 13 labelled cases is **Recall@4 0.90**, **MRR 0.85**, and
  **unsupported-query empty-result accuracy 0.6667**.

The imperfect retrieval result is intentional evidence, not a marketing omission. It identifies a
missed damaged-item paraphrase and a printer false positive. The optional hybrid path must improve
those cases without reducing empty-result accuracy before it is selected.

The measured local `all-minilm` semantic-gated hybrid does so on the same fixed set: Recall@4
**1.00**, MRR **0.95**, and unsupported-query empty-result accuracy **1.00**. Its p50 latency is
**61.871 ms**, so BM25 remains the no-service, sub-millisecond default and hybrid is an explicit
quality/latency choice.

## What is implemented, and what is not claimed

Implemented and locally verified: deterministic provider workflow, SQLite persistence, tenant
isolation, BM25 retrieval, citation/action gates, metrics, evaluation, the failure demonstration,
and all tests above.

Implemented but not called against a paid service locally: the Anthropic provider adapter. The CI
contract and local stand-in cover its typed boundary, not real-model answer quality.

Implemented and locally benchmarked as an optional experiment: local Ollama dense retrieval,
semantic out-of-domain gating, and hybrid rank fusion. PostgreSQL is exercised in CI; a durable
pgvector index and production traffic are not claimed.

## Why this matters to a client

The deliverable is not merely an LLM prompt. It shows where customer data is partitioned, what the
model can and cannot decide, how a reviewer reconstructs a run, which failures trigger operational
work, and which measured weakness should receive the next engineering dollar. The same boundaries
apply to ticketing, CRM, onboarding, and other AI-assisted business workflows.
