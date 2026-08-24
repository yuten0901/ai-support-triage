"""Retry delays.

Deterministic given a jitter source, so the schedule is a table test rather
than an approximation. Two properties matter and both are asserted in
``tests/unit/test_backoff.py``:

* the delay is bounded -- an unbounded exponential eventually means "never",
  which reads as a hang rather than a failure;
* a provider-supplied ``Retry-After`` wins over our own schedule, but is still
  clamped. A provider asking us to wait an hour is not something a request
  handler can honour, and pretending to is worse than failing fast.

Full jitter (uniform over ``[0, delay]``) rather than fixed backoff, because
the failure mode being defended against is a fleet of callers retrying in
lockstep after the same outage.
"""

from __future__ import annotations

import random
from collections.abc import Callable

#: Injected so tests can pin it. `random.random` in production.
JitterSource = Callable[[], float]


def compute_delay(
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
    retry_after: float | None = None,
    jitter: JitterSource = random.random,  # noqa: S311 - not cryptographic
) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based).

    ``retry_after`` is the provider's own hint, in seconds. It is honoured
    when present, clamped to ``max_seconds``, and *not* jittered -- the whole
    point of the hint is that the provider named a specific time.
    """
    if attempt < 1:
        raise ValueError("attempt is 1-based")

    if retry_after is not None:
        return max(0.0, min(retry_after, max_seconds))

    uncapped = base_seconds * (2 ** (attempt - 1))
    return float(min(uncapped, max_seconds) * jitter())
