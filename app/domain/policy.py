"""The rules that are deliberately *not* the model's job.

Three kinds of decision live here, and all three are pure functions of their
arguments so the whole set is covered by tables in
``tests/unit/test_policy.py``:

1. **The pre-check** -- can this ticket be answered without calling a model at
   all? An out-of-office bounce does not need an LLM to recognise, and calling
   one anyway is a cost with no upside.

2. **The injection scan** -- does the untrusted text contain something shaped
   like an instruction to us? This is a heuristic and is documented as one; it
   does not block anything by itself. Its only effect is to make a write action
   in the same run require a human.

3. **The outcome gate** -- given a model's proposal, what actually happens.
   This is where "the model said to refund $400" becomes "a person will decide
   whether to refund $400". The model has no field it can set to skip this.

The ordering inside :func:`decide_outcome` is load-bearing and is asserted
directly: a refusal is reported as a refusal even if confidence was also low,
and missing evidence is reported as missing evidence rather than as a
low-confidence escalation, because those tell an operator to do different
things (fix the model prompt vs. fix the knowledge base).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ai.schemas import Resolution
from app.domain.evidence import EvidenceSet
from app.domain.states import WRITE_ACTIONS, ActionKind, EscalationReason, TriageStatus

# ---------------------------------------------------------------------------
# 1. Pre-check: work a model should never be paid to do
# ---------------------------------------------------------------------------

#: Phrases that identify machine-generated mail. Matched case-insensitively
#: against the whole ticket. Deliberately conservative: a false positive here
#: silently drops a real customer request, which is far worse than the price of
#: one model call.
_AUTO_REPLY_MARKERS = (
    "out of office",
    "automatic reply",
    "auto-reply",
    "autoreply",
    "delivery status notification",
    "undeliverable:",
    "this is an automated message",
    "do not reply to this email",
)

#: A closing courtesy with no request in it.
_COURTESY_ONLY = re.compile(
    r"^\W*(thanks|thank you|thx|cheers|much appreciated|great, thanks|perfect, thanks)"
    r"[\s!.,\-]*(again|so much|a lot)?[\s!.,\-]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PrecheckResult:
    """Whether the deterministic layer can answer without a model."""

    handled: bool
    reason: str | None = None
    note: str | None = None


def precheck(subject: str, body: str) -> PrecheckResult:
    """Decide whether this ticket needs an LLM at all."""
    haystack = f"{subject}\n{body}".casefold()

    for marker in _AUTO_REPLY_MARKERS:
        if marker in haystack:
            return PrecheckResult(
                handled=True,
                reason="automated_message",
                note=f"Matched auto-reply marker {marker!r}; no customer request present.",
            )

    stripped = body.strip()
    if _COURTESY_ONLY.match(stripped):
        return PrecheckResult(
            handled=True,
            reason="courtesy_closing",
            note="Message is a closing courtesy with no request.",
        )

    return PrecheckResult(handled=False)


# ---------------------------------------------------------------------------
# 2. Injection scan over untrusted text
# ---------------------------------------------------------------------------

#: Patterns that look like an attempt to address the *system* rather than the
#: support agent. This is pattern matching, not security: a determined attacker
#: will phrase it differently. What it buys is that the common, low-effort case
#: is visible in the run trace and forces a human onto any write action.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_instructions", re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I)),
    ("override_instructions", re.compile(r"\bdisregard\s+(the\s+)?(above|previous|system|earlier)\b", re.I)),
    ("role_reassignment", re.compile(r"\byou\s+are\s+now\s+(a|an|the)\b", re.I)),
    ("system_prompt_probe", re.compile(r"\b(system|developer)\s+prompt\b", re.I)),
    ("instruction_extraction", re.compile(r"\b(reveal|print|show|repeat)\s+(me\s+)?your\s+(instructions?|prompt|rules)\b", re.I)),
    ("policy_override_claim", re.compile(r"\b(policy|rules?)\s+(does\s+not|do\s+not|doesn't|don't)\s+apply\b", re.I)),
    ("forged_authority", re.compile(r"\b(as|this\s+is)\s+(the\s+)?(admin|administrator|ceo|支店長)\b", re.I)),
    ("delimiter_injection", re.compile(r"</?(system|developer|instructions?)>", re.I)),
)


@dataclass(frozen=True, slots=True)
class InjectionScan:
    """Result of scanning one piece of untrusted text."""

    suspected: bool
    patterns: tuple[str, ...] = ()


def scan_for_injection(*texts: str) -> InjectionScan:
    """Look for instruction-shaped content in text we did not author."""
    hits: list[str] = []
    for text in texts:
        for name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text) and name not in hits:
                hits.append(name)
    return InjectionScan(suspected=bool(hits), patterns=tuple(hits))


# ---------------------------------------------------------------------------
# 3. The outcome gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomePolicy:
    """The business thresholds, passed in rather than read from settings.

    Passing them in is what lets a test state the threshold it is exercising
    instead of mutating global configuration -- and it is why these rules can be
    covered exhaustively from a table.
    """

    auto_refund_cap_minor: int
    min_auto_confidence: float
    high_risk_categories: frozenset[str]


@dataclass(slots=True)
class Outcome:
    """What the system decided, and why."""

    status: TriageStatus
    escalation_reasons: list[EscalationReason] = field(default_factory=list)
    #: The action to execute, if any. ``None`` when nothing should happen now --
    #: either because there is nothing to do, or because a human has to approve
    #: it first. Which of those it is, is carried by ``status``.
    action: ActionKind | None = None
    rationale: str = ""


def decide_outcome(
    resolution: Resolution,
    *,
    category: str,
    model_confidence: float,
    evidence: EvidenceSet,
    tool_failure_count: int,
    injection: InjectionScan,
    account_plan_known: bool,
    policy: OutcomePolicy,
) -> Outcome:
    """Turn a model proposal into a decision.

    ``model_confidence`` is passed separately from ``resolution.confidence``
    because the run has two confidences (classification and resolution) and the
    gate uses the lower of the two; taking the maximum would let a confident
    answer to a misclassified ticket sail through.
    """
    # --- Non-answers first. These are outcomes, not degraded successes. ----
    if resolution.answer_status == "refused":
        return Outcome(
            status=TriageStatus.REJECTED_BY_MODEL,
            rationale=resolution.escalation_reason or "The model declined to answer this request.",
        )

    if resolution.answer_status == "insufficient_evidence" or not evidence:
        return Outcome(
            status=TriageStatus.INSUFFICIENT_EVIDENCE,
            rationale=(
                "No retrieved policy section supports an answer to this request. "
                "This is a knowledge-base gap, not a model failure."
            ),
        )

    # --- Escalation gates. All are evaluated; none short-circuits, because a
    # reviewer benefits from seeing every reason at once. -------------------
    reasons: list[EscalationReason] = []
    action = resolution.recommended_action

    if min(model_confidence, resolution.confidence) < policy.min_auto_confidence:
        reasons.append(EscalationReason.LOW_CONFIDENCE)

    if resolution.risk == "high":
        reasons.append(EscalationReason.POLICY_REQUIRES_APPROVAL)

    if action.kind in WRITE_ACTIONS and category in policy.high_risk_categories:
        reasons.append(EscalationReason.POLICY_REQUIRES_APPROVAL)

    if action.kind is ActionKind.ISSUE_REFUND:
        amount = action.amount_minor or 0
        if amount >= policy.auto_refund_cap_minor:
            reasons.append(EscalationReason.HIGH_VALUE_ACTION)

    # Two policy sections that address different audiences, and no tool result
    # telling us which audience this customer is in, is a genuine ambiguity.
    # Answering it would mean picking one at random.
    if len(evidence.audiences) > 1 and not account_plan_known:
        reasons.append(EscalationReason.CONFLICTING_EVIDENCE)

    if tool_failure_count >= 2:
        reasons.append(EscalationReason.REPEATED_TOOL_FAILURE)

    # Injection alone does not escalate -- a customer quoting a phishing email
    # at us is not an incident. Injection *plus a write action* does.
    if injection.suspected and action.kind in WRITE_ACTIONS:
        reasons.append(EscalationReason.SUSPECTED_PROMPT_INJECTION)

    if resolution.escalation_requested:
        reasons.append(EscalationReason.MODEL_REQUESTED_ESCALATION)

    if reasons:
        deduped: list[EscalationReason] = []
        for reason in reasons:
            if reason not in deduped:
                deduped.append(reason)
        return Outcome(
            status=TriageStatus.NEEDS_HUMAN_REVIEW,
            escalation_reasons=deduped,
            action=action.kind if action.kind in WRITE_ACTIONS else None,
            rationale="Held for review: " + ", ".join(r.value for r in deduped) + ".",
        )

    return Outcome(
        status=TriageStatus.AUTO_RESOLVED,
        action=action.kind if action.kind in WRITE_ACTIONS else None,
        rationale=action.justification or "Cleared by policy for automatic handling.",
    )
