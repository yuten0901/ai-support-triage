"""Public API contracts."""

from datetime import datetime

from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=20_000)
    customer_id: str | None = Field(default=None, max_length=128)


class ReviewDecision(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    decision: str = Field(pattern="^(approved|rejected|modified)$")
    note: str | None = Field(default=None, max_length=2_000)
    final_reply: str | None = Field(default=None, max_length=20_000)


class RunSummary(BaseModel):
    run_id: str
    status: str
    category: str | None
    urgency: str | None
    confidence: float | None
    answer_status: str | None
    reply_draft: str | None
    recommended_action: str | None
    action_executed: bool
    escalation_reasons: list[str]
    rationale: str | None
    error_kind: str | None
    provider_call_count: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    latency_ms: int
    started_at: datetime
    completed_at: datetime | None
