"""The vocabulary of outcomes.

This module exists because the single most common way an LLM-backed system
lies to its operators is by flattening distinct outcomes into one bucket --
usually "it worked" or "error". The brief for this service treats the
following as *four different things*, and so does every layer below:

* a decision the system made automatically,
* a decision the system deliberately handed to a human,
* an honest "the knowledge base does not answer this",
* and a system failure.

Plus two more that are easy to lose: a valid "nothing to do here" (an
auto-reply, a thank-you note) and a model that declined to answer.

Everything downstream -- the API response, the database, the log line, the
usage summary, the evaluation report -- keys off :class:`TriageStatus`, so
these six can never be silently merged. :func:`is_terminal_success` is the only
place that groups them, and it groups them for *one* purpose (did the workflow
complete without a system fault), not for reporting.
"""

from __future__ import annotations

from enum import StrEnum


class TriageStatus(StrEnum):
    """How a triage run ended. Mutually exclusive, and never inferred."""

    #: The system decided and (if an action was warranted) executed it.
    AUTO_RESOLVED = "auto_resolved"

    #: Valid outcome: the ticket needs no action (auto-reply, closing thanks).
    NO_ACTION_REQUIRED = "no_action_required"

    #: Retrieval found nothing that supports an answer. Not an error, and
    #: deliberately not the same as "needs human review" -- there is no useful
    #: question to put in front of a reviewer, the knowledge base is the gap.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    #: The model produced a usable answer, but policy routes it to a person.
    NEEDS_HUMAN_REVIEW = "needs_human_review"

    #: The model declined to answer (safety, out of scope). Distinct from a
    #: system error: nothing is broken, and retrying will not help.
    REJECTED_BY_MODEL = "rejected_by_model"

    #: A system fault: provider outage, exhausted retries, unrepairable output,
    #: budget or deadline exceeded. The only status that means "page someone".
    FAILED = "failed"


#: Statuses that represent the workflow completing as designed. Used for
#: alerting thresholds, never for reporting to a user -- a reader who is told
#: "successful" when the real answer was INSUFFICIENT_EVIDENCE has been misled.
_NON_FAULT = frozenset(
    {
        TriageStatus.AUTO_RESOLVED,
        TriageStatus.NO_ACTION_REQUIRED,
        TriageStatus.INSUFFICIENT_EVIDENCE,
        TriageStatus.NEEDS_HUMAN_REVIEW,
        TriageStatus.REJECTED_BY_MODEL,
    }
)


def is_terminal_success(status: TriageStatus) -> bool:
    """Whether the run finished without a *system* fault."""
    return status in _NON_FAULT


class ErrorKind(StrEnum):
    """Why a run ended in :attr:`TriageStatus.FAILED`.

    Split by what an operator would do about it: the provider ones are
    infrastructure, ``INVALID_MODEL_OUTPUT`` and ``UNGROUNDED_CITATION`` are
    prompt or model regressions, and the budget ones are configuration.
    """

    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_BAD_REQUEST = "provider_bad_request"
    #: Output could not be parsed or did not satisfy the schema, after repair.
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    #: Output was schema-valid but cited evidence that was never shown to it,
    #: or quoted text that does not appear in the cited chunk.
    UNGROUNDED_CITATION = "ungrounded_citation"
    #: A tool failed in a way that is our problem, not the model's.
    TOOL_INFRASTRUCTURE = "tool_infrastructure"
    BUDGET_EXCEEDED = "budget_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL = "internal"


class EscalationReason(StrEnum):
    """Why a run was routed to a human.

    A run can carry several. They are recorded as a list rather than collapsed
    to the "first" one, because "low confidence *and* suspected injection" is a
    materially different thing for a reviewer to see than either alone.
    """

    LOW_CONFIDENCE = "low_confidence"
    HIGH_VALUE_ACTION = "high_value_action"
    POLICY_REQUIRES_APPROVAL = "policy_requires_approval"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    REPEATED_TOOL_FAILURE = "repeated_tool_failure"
    SUSPECTED_PROMPT_INJECTION = "suspected_prompt_injection"
    MODEL_REQUESTED_ESCALATION = "model_requested_escalation"


class StepKind(StrEnum):
    """The stages of the workflow, in the order they can occur.

    Recorded per run so that "why did the system produce this result?" is
    answered by reading a table rather than by re-running anything.
    """

    PRECHECK = "precheck"
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    PLAN_TOOLS = "plan_tools"
    EXECUTE_TOOL = "execute_tool"
    RESOLVE = "resolve"
    DECIDE = "decide"
    EXECUTE_ACTION = "execute_action"


class StepStatus(StrEnum):
    OK = "ok"
    #: The step ran, and its outcome was a deliberate non-answer.
    DECLINED = "declined"
    FAILED = "failed"
    #: The step was skipped by design (e.g. no tool needed).
    SKIPPED = "skipped"


class ReviewState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class ActionKind(StrEnum):
    """What the system proposes to *do*, as opposed to what it says.

    Kept deliberately small. The model may only propose one of these; it has no
    way to express an action the tool registry cannot execute.
    """

    NONE = "none"
    REPLY_ONLY = "reply_only"
    ISSUE_REFUND = "issue_refund"
    CREATE_ESCALATION = "create_escalation"


#: Actions that change state somewhere outside this service. Every one of them
#: goes through the approval gate in :mod:`app.domain.policy`.
WRITE_ACTIONS = frozenset({ActionKind.ISSUE_REFUND, ActionKind.CREATE_ESCALATION})
