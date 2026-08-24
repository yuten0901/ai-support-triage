"""What a run leaves behind.

The tables here exist to answer one question after the fact: *why did the
system produce this result?* Everything recorded is something a person
debugging a bad decision would otherwise have to guess at -- which chunks the
model saw, which of them it cited, what each provider call cost, how many
attempts it took, and which deterministic rule made the final call.

Two shapes are deliberate:

**Token usage is nullable, and null does not mean zero.** ``input_tokens`` is
``None`` when the provider did not report usage. A zero there would silently
understate the usage summary and there would be no way to tell the difference
later. The same applies to ``estimated_cost_usd``: a model with no entry in the
price table yields ``None``, not ``0.0``.

**Evidence rows record ``was_cited``.** Retrieval and citation are separate
facts. "We retrieved four chunks and the model cited one of them" and "we
retrieved one chunk" produce different rows, and only the first tells you the
retrieval was wider than the answer needed.

Nothing here stores a prompt body or raw provider payload. Those contain the
customer's text, and a debugging table is the wrong place for it -- the run's
inputs are already in ``tickets``, once, where a deletion request can reach
them.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for every table in the service."""


def _utcnow_column() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False)


class Ticket(Base):
    """The inbound request, stored once.

    ``external_id`` is the caller's own reference and is unique: re-submitting
    the same ticket returns the existing run rather than paying for a second
    one. That is cost control, not just tidiness -- a retrying client should not
    be able to multiply the model bill.
    """

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    subject: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    received_at: Mapped[datetime] = _utcnow_column()

    runs: Mapped[list[TriageRun]] = relationship(back_populates="ticket")


class TriageRun(Base):
    """One pass of the workflow over one ticket."""

    __tablename__ = "triage_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticket_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )

    #: One of :class:`app.domain.states.TriageStatus`. Stored as its string
    #: value rather than a database enum: adding an outcome should be a code
    #: change, not a migration, and the six values are validated on the way in.
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Set only when ``status == 'failed'``. See ``ErrorKind``.
    error_kind: Mapped[str | None] = mapped_column(String(48), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    category: Mapped[str | None] = mapped_column(String(48), nullable=True)
    urgency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: The lower of the classification and resolution confidences -- the value
    #: the policy gate actually used, not an average of the two.
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reply_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action_executed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Comma-separated ``EscalationReason`` values, in the order they were
    #: found. Empty string means "none", which is different from a run that was
    #: never gated at all (those have ``status != needs_human_review``).
    escalation_reasons: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Identifies the exact set of prompt templates this run used. A prompt
    #: edit changes this string, so a behaviour change can be tied to a prompt
    #: change without diffing anything.
    prompt_bundle_version: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)

    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime] = _utcnow_column()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    ticket: Mapped[Ticket] = relationship(back_populates="runs")
    steps: Mapped[list[RunStep]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunStep.seq"
    )
    llm_calls: Mapped[list[LLMCall]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="LLMCall.seq"
    )
    evidence: Mapped[list[EvidenceRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EvidenceRecord.rank"
    )
    tool_calls: Mapped[list[ToolCallRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="ToolCallRecord.seq"
    )
    review: Mapped[Review | None] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )


Index("ix_triage_runs_ticket", TriageRun.ticket_id)
Index("ix_triage_runs_status", TriageRun.status)


class RunStep(Base):
    """One stage of the state machine, in the order it ran.

    A skipped step is recorded, not omitted. "The tool step did not run because
    no tool was needed" and "the tool step is missing from the trace" look
    identical if only successful steps are written down.
    """

    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("triage_runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[TriageRun] = relationship(back_populates="steps")

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_step_seq"),)


class LLMCall(Base):
    """One HTTP round trip to the provider.

    Retries and repairs are separate rows rather than a counter, because
    "three attempts" hides whether the provider timed out three times or
    answered three times with invalid JSON -- two problems with two different
    owners.
    """

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("triage_runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    step_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    #: ``initial``, ``transport_retry`` or ``repair``. The distinction is the
    #: whole point of the two separate attempt budgets in the settings.
    attempt_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    #: ``ok``, ``invalid_output``, or a ``ProviderError`` class name.
    outcome: Mapped[str] = mapped_column(String(48), nullable=False)
    #: Provider-side id. Null when the provider did not return one.
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Null when the provider did not report usage. Not zero -- see module docstring.
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[TriageRun] = relationship(back_populates="llm_calls")

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_llm_call_seq"),)


class EvidenceRecord(Base):
    """A chunk that was shown to the model, and whether it ended up cited."""

    __tablename__ = "evidence_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("triage_runs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(String(160), nullable=False)
    document_id: Mapped[str] = mapped_column(String(96), nullable=False)
    document_title: Mapped[str] = mapped_column(String(256), nullable=False)
    document_version: Mapped[str] = mapped_column(String(32), nullable=False)
    heading: Mapped[str] = mapped_column(String(256), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    was_cited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: The span the model quoted, when it cited this chunk. Stored because a
    #: citation that passed validation is the strongest evidence that the reply
    #: is grounded, and a reviewer should be able to read it without re-running.
    cited_quote: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[TriageRun] = relationship(back_populates="evidence")


class ToolCallRecord(Base):
    """One attempted tool invocation.

    ``idempotency_key`` is recorded for write actions so a duplicate execution
    is visible in the table rather than only in the downstream system.
    """

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("triage_runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Validated arguments, re-serialised from the tool's own model. Never the
    #: raw string the model produced -- that may not be valid JSON at all.
    arguments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: ``ok``, ``invalid_arguments``, ``not_found``, ``failed``.
    #: ``not_found`` is a *successful* call with a negative answer.
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    run: Mapped[TriageRun] = relationship(back_populates="tool_calls")

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_tool_call_seq"),)


class Review(Base):
    """The human side of an escalated run.

    A row exists only for runs that were actually routed to a person, so
    "pending review" is a fact about the queue rather than a flag on every run.
    ``decided_at`` and ``reviewer`` are recorded together: an approval with no
    reviewer is not an approval.
    """

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("triage_runs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    reasons: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Set when the reviewer edited the draft rather than accepting it. The
    #: gap between this and ``TriageRun.reply_draft`` is the most direct signal
    #: available of how good the model's drafts actually are.
    final_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _utcnow_column()
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[TriageRun] = relationship(back_populates="review")


Index("ix_reviews_state", Review.state)
