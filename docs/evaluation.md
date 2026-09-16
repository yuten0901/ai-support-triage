# Evaluation

`evals/cases.json` contains eight deterministic scenarios: eligible and expired refunds, delayed
shipping, account access, subscription changes, courtesy mail, an unsupported question, and a
prompt-injection-shaped request. `python -m evals.runner` checks expected category/status and
re-runs grounding validation, then writes `reports/eval-report.json`.

The current offline run passed 8/8 cases. This is a regression set, not a statistical claim about
production quality. It intentionally reports exact case count and provider mode.

## Retrieval benchmark

`evals/retrieval_cases.json` separately measures retrieval rather than hiding it inside end-to-end
status checks. Ten answerable cases include vocabulary-aligned and paraphrased questions; three
unsupported cases test whether the retriever returns an honest empty set. The runner reports
Recall@k, MRR, empty-result accuracy, and local latency percentiles.

The checked-in BM25 snapshot (`reports/retrieval-bm25.json`) is Recall@4 **0.90**, MRR **0.85**, and
empty-result accuracy **0.6667**. It misses the paraphrased damaged-item case and returns a false
positive for the unrelated printer question. Those failures are retained as the acceptance target
for the optional hybrid path, rather than editing the dataset to make the baseline look perfect.

Run the baseline with `python -m evals.retrieval_runner`; its volatile latest report is written under
the ignored `evals/reports/` directory so latency changes do not dirty the repository. A local
Ollama instance with the requested embedding model enables `python -m evals.retrieval_runner
--mode hybrid --output evals/reports/retrieval-hybrid.json`. A hybrid result is not a release claim
until it improves the target failure without reducing unsupported-query accuracy.

`scripts/verify_mutations.py` copies the repository to a temporary directory, seeds four defects,
and runs the relevant test for each. A successful mutation run means all four defects were caught:
strict schema disabled, unknown citation accepted, tool argument validation bypassed, and one
extra transport retry permitted.

Real Anthropic quality and latency were not measured because no API credential was available.
Ollama hybrid quality has not yet been measured on this machine. PostgreSQL was not available
locally; the workflow definition runs that backend in CI.
