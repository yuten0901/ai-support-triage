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
