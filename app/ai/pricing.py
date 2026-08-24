"""Turning token counts into dollars, and being honest when it cannot.

A cost figure that is quietly wrong is worse than no cost figure, because
somebody will put it on a dashboard. So there are three distinct states here,
and they are never merged:

* **A known cost.** The model is in the table and the provider reported usage.
* **Genuinely zero.** The in-process fake provider costs nothing to run. That
  is a real measured zero, not a missing value, and it has a table entry
  saying so.
* **Unknown.** The model is not in the table, or the provider did not report
  usage. This returns ``None``, and ``None`` propagates all the way to the API
  response and the usage summary rather than being coerced to ``0.0``
  somewhere in the middle.

The third case has a consequence worth stating: **the per-run cost budget
cannot be enforced against an unknown price.** The service does not pretend
otherwise -- :func:`is_priced` is checked at startup, and a run whose model has
no price records that its dollar budget was unenforceable. The call-count and
deadline limits still apply, so the run is still bounded; it is only the
*money* ceiling that is unavailable, and saying so is better than enforcing a
budget against a fabricated zero.

Prices are USD per million tokens, current as of 2026-08. They are a local copy
of a vendor's public price list and will drift; ``docs/observability.md`` says
where to check them.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.provider import TokenUsage


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    # Not a placeholder. The fake provider runs in this process and makes no
    # network call, so its cost is exactly zero -- a measured value, entered
    # deliberately so the demo's usage summary reads "$0.0000" rather than
    # "unknown" and the difference stays meaningful.
    "fake-triage-v1": ModelPrice(0.0, 0.0),
}


def is_priced(model: str) -> bool:
    """Whether a dollar figure can be produced for this model at all."""
    return model in PRICES


def estimate_cost(model: str, usage: TokenUsage) -> float | None:
    """Cost of one call, or ``None`` when it cannot be known.

    ``None`` for an unpriced model *and* for usage the provider declined to
    report -- in both cases the honest answer is that we do not know, and the
    caller is expected to carry that through rather than substitute a number.
    """
    price = PRICES.get(model)
    if price is None or not usage.reported:
        return None
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000


def worst_case_cost(
    model: str, *, estimated_input_tokens: int, max_output_tokens: int
) -> float | None:
    """Upper bound on what a call *about* to be made could cost.

    Used before dialling out, so a run can be stopped before it overspends
    rather than after. ``estimated_input_tokens`` is a rough character-count
    heuristic -- the exact number is only known once the provider answers, and
    by then the money is spent. Erring high is the correct direction for a
    ceiling.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        estimated_input_tokens * price.input_per_mtok + max_output_tokens * price.output_per_mtok
    ) / 1_000_000


#: Characters per token. A crude average that is close enough for a budget
#: ceiling and is documented as crude rather than dressed up as a tokenizer.
CHARS_PER_TOKEN = 4


def tokens_from_chars(char_count: int) -> int:
    """Rough token count for budgeting only. Never recorded as actual usage."""
    return max(1, char_count // CHARS_PER_TOKEN)
