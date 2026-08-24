# Architecture

The service is an explicit pipeline: precheck, classify, retrieve, plan tools, execute read-only
tools, resolve, verify grounding, decide, and optionally execute an approved action. Every loop has
a configured ceiling. The model proposes; deterministic code validates and decides.

The provider contract accepts messages and a JSON schema. The fake provider uses the same prompt
and schema boundary as Anthropic, keeping normal tests deterministic without replacing the
orchestrator with fixture lookups. Provider transport errors, refusals, malformed output, semantic
grounding failures, and empty retrieval each retain distinct types and terminal states.

Policy documents are Markdown with versioned chunk identifiers. BM25 is appropriate for this
small, vocabulary-aligned corpus and makes retrieval inspectable. A citation passes only when its
chunk was retrieved for that run and its normalized quote occurs verbatim in that chunk.

SQLite is the local default. SQLAlchemy confines backend differences to engine construction, and
the CI matrix includes PostgreSQL. Each run persists its state-machine steps, provider attempts,
retrieved and cited evidence, validated tool arguments, outcome, cost, and review record.
