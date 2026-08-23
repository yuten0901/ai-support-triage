"""Write actions -- the side of the system the model cannot reach.

These are deliberately *not* in the tool registry. The registry is what the
model may request; this module is what the deterministic policy layer may do
once it has decided. Keeping them in separate modules means the separation is
visible rather than being a convention someone has to remember: there is no
code path from a model output to a function here that does not pass through
:func:`app.domain.policy.decide_outcome`.

Every action carries an idempotency key derived from the run and the action.
The realistic failure this defends against is a retried request re-running the
workflow and issuing a second refund for the same ticket -- one of the few ways
an AI integration can lose real money.

**Scope, stated plainly:** the ledger below is in-process. Durable,
crash-safe idempotency across restarts is a database problem, thoroughly
covered by the sibling `order-sync-service` project, and duplicating it here
would add a lot of code that has nothing to do with the subject of this one.
What is demonstrated here is the *boundary* -- that a model-proposed action is
gated, keyed and recorded -- not a production payments integration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Protocol

ActionStatus = Literal["executed", "duplicate_ignored", "failed"]


@dataclass(frozen=True, slots=True)
class ActionResult:
    status: ActionStatus
    reference: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """A duplicate is a success: the intended effect already exists."""
        return self.status in ("executed", "duplicate_ignored")


class ActionExecutionError(Exception):
    """The downstream system rejected or could not process the action."""


def idempotency_key(run_id: str, action: str, target: str, amount_minor: int | None) -> str:
    """Stable key for one intended effect.

    Derived from the run rather than generated, so the *same* run retried
    produces the same key and cannot double-execute, while a genuinely new run
    against the same order produces a different one -- a customer can be
    refunded twice for two separate problems, which is correct.
    """
    material = f"{run_id}|{action}|{target}|{amount_minor if amount_minor is not None else '-'}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class ActionExecutor(Protocol):
    """What the workflow needs in order to carry out an approved action."""

    def issue_refund(
        self, *, order_id: str, amount_minor: int, currency: str, key: str
    ) -> ActionResult: ...

    def create_escalation(
        self, *, run_id: str, reason: str, summary: str, key: str
    ) -> ActionResult: ...


@dataclass
class LedgerExecutor:
    """Records actions against an in-process ledger.

    Stands in for a payments API and a ticketing system. It is a stub in what
    it talks to and not in what it enforces: the key check is real, and a test
    that executes the same action twice sees ``duplicate_ignored``.
    """

    refunds: dict[str, dict[str, object]] = field(default_factory=dict)
    escalations: dict[str, dict[str, object]] = field(default_factory=dict)
    _sequence: int = 0

    def _next_reference(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence:06d}"

    def issue_refund(
        self, *, order_id: str, amount_minor: int, currency: str, key: str
    ) -> ActionResult:
        if amount_minor <= 0:
            # Reached only if the policy layer passed something impossible;
            # refusing here rather than recording a zero refund keeps the
            # ledger honest about what actually happened.
            raise ActionExecutionError("refund amount must be positive")

        existing = self.refunds.get(key)
        if existing is not None:
            return ActionResult(
                status="duplicate_ignored",
                reference=str(existing["reference"]),
                detail="A refund with this idempotency key was already issued.",
            )

        reference = self._next_reference("RFND")
        self.refunds[key] = {
            "reference": reference,
            "order_id": order_id,
            "amount_minor": amount_minor,
            "currency": currency,
        }
        return ActionResult(
            status="executed",
            reference=reference,
            detail=f"Refunded {amount_minor} {currency} against {order_id}.",
        )

    def create_escalation(
        self, *, run_id: str, reason: str, summary: str, key: str
    ) -> ActionResult:
        existing = self.escalations.get(key)
        if existing is not None:
            return ActionResult(
                status="duplicate_ignored",
                reference=str(existing["reference"]),
                detail="An escalation with this idempotency key already exists.",
            )

        reference = self._next_reference("ESC")
        self.escalations[key] = {
            "reference": reference,
            "run_id": run_id,
            "reason": reason,
            "summary": summary,
        }
        return ActionResult(
            status="executed", reference=reference, detail=f"Escalation {reference} opened."
        )
