# Evaluation

`evals/cases.json` contains eight deterministic scenarios: eligible and expired refunds, delayed
shipping, account access, subscription changes, courtesy mail, an unsupported question, and a
prompt-injection-shaped request. `python -m evals.runner` checks expected category/status and
re-runs grounding validation, then writes `reports/eval-report.json`.

The current offline run passed 8/8 cases. This is a regression set, not a statistical claim about
production quality. It intentionally reports exact case count and provider mode.

`scripts/verify_mutations.py` copies the repository to a temporary directory, seeds four defects,
and runs the relevant test for each. A successful mutation run means all four defects were caught:
strict schema disabled, unknown citation accepted, tool argument validation bypassed, and one
extra transport retry permitted.

Real Anthropic quality and latency were not measured because no API credential was available.
PostgreSQL was not available locally; the workflow definition runs that backend in CI.
