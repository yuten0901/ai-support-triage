"""Time as an injected dependency.

Retry schedules, run deadlines and latency measurements are all assertions the
test suite needs to make exactly. With a real clock they become races; with an
injected one they become table tests, and the suite finishes in seconds without
a single ``sleep``.

Two separate concerns live here on purpose:

* ``now()`` is wall-clock time, used for timestamps that a human reads.
* ``monotonic()`` is elapsed time, used for deadlines and latency. It cannot go
  backwards when the host's clock is adjusted, which is exactly the property a
  deadline needs and a timestamp does not.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Everything in this service that needs time asks for one of these."""

    def now(self) -> datetime:
        """Current wall-clock time, timezone-aware, in UTC."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin. Only differences are meaningful."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds``. Retry backoff is the only caller."""
        ...


class SystemClock:
    """The production clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FrozenClock:
    """A clock that only moves when a test moves it.

    ``sleep`` advances the clock instead of blocking, so a test can assert the
    *total* delay a retry policy imposed (``clock.slept``) without waiting for
    it, and code under test still observes time passing across the sleep.
    """

    def __init__(self, start: datetime, monotonic_start: float = 1_000.0) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock needs a timezone-aware datetime")
        self._now = start
        self._monotonic = monotonic_start
        #: Every duration passed to :meth:`sleep`, in order.
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._now = self._now + timedelta(seconds=seconds)
        self._monotonic += seconds
