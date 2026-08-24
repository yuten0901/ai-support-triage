"""Application assembly and persistence for triage runs."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.ai.anthropic_provider import AnthropicProvider
from app.ai.client import RunLimits
from app.ai.fake_provider import FakeProvider
from app.ai.provider import LLMProvider
from app.config import Settings
from app.db.models import (
    EvidenceRecord,
    LLMCall,
    Review,
    RunStep,
    Ticket,
    ToolCallRecord,
    TriageRun,
)
from app.domain.clock import SystemClock
from app.domain.policy import OutcomePolicy
from app.domain.states import TriageStatus
from app.rag.index import KnowledgeIndex
from app.rag.loader import load_documents
from app.tools.actions import LedgerExecutor
from app.tools.builtin import RefundRules, build_registry
from app.tools.store import JsonFileStore
from app.workflow.orchestrator import Orchestrator, OrchestratorConfig, RunResult, TriageRequest


@dataclass(slots=True)
class Services:
    settings: Settings
    sessions: sessionmaker[Session]
    orchestrator: Orchestrator
    index: KnowledgeIndex


def build_services(settings: Settings, sessions: sessionmaker[Session]) -> Services:
    """Construct the deterministic default service graph."""
    root = Path.cwd()
    clock = SystemClock()
    index = KnowledgeIndex(load_documents(root / settings.knowledge_dir))
    store = JsonFileStore(root / "data", clock)
    tools = build_registry(
        store,
        clock,
        RefundRules(
            standard_window_days=settings.refund_window_days,
            enterprise_window_days=max(settings.refund_window_days, 60),
        ),
    )
    provider: LLMProvider
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        provider = AnthropicProvider(
            api_key=settings.anthropic_api_key.get_secret_value(),
            base_url=settings.anthropic_base_url,
            default_timeout_seconds=settings.llm_request_timeout_seconds,
        )
    else:
        provider = FakeProvider()
    limits = RunLimits(
        max_transport_attempts=settings.llm_max_transport_attempts,
        max_repair_attempts=settings.llm_max_repair_attempts,
        max_provider_calls_per_step=settings.llm_max_provider_calls_per_step,
        retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.llm_retry_max_delay_seconds,
        request_timeout_seconds=settings.llm_request_timeout_seconds,
        run_deadline_seconds=settings.run_deadline_seconds,
        cost_budget_usd=settings.run_cost_budget_usd,
        max_output_tokens=settings.llm_max_output_tokens,
        effort=settings.llm_effort,
        use_structured_outputs=settings.llm_use_structured_outputs,
    )
    config = OrchestratorConfig(
        limits=limits,
        policy=OutcomePolicy(
            auto_refund_cap_minor=settings.auto_refund_cap_minor,
            min_auto_confidence=settings.min_auto_confidence,
            high_risk_categories=frozenset(settings.high_risk_categories),
        ),
        max_tool_calls_per_run=settings.max_tool_calls_per_run,
        max_tool_attempts=settings.max_tool_attempts,
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_min_score=settings.retrieval_min_score,
        max_ticket_body_chars=settings.max_ticket_body_chars,
        model=settings.llm_model,
    )
    orchestrator = Orchestrator(
        provider=provider,
        index=index,
        tools=tools,
        actions=LedgerExecutor(),
        clock=clock,
        jitter=random.random,
        config=config,
    )
    return Services(settings=settings, sessions=sessions, orchestrator=orchestrator, index=index)


def execute_and_record(services: Services, request: TriageRequest) -> TriageRun:
    """Return an existing idempotent run, or execute and persist a new one."""
    with services.sessions.begin() as session:
        existing = session.scalar(select(Ticket).where(Ticket.external_id == request.external_id))
        if existing and existing.runs:
            return existing.runs[-1]

    result = services.orchestrator.run(request)
    with services.sessions.begin() as session:
        ticket = Ticket(
            id=uuid.uuid4().hex,
            external_id=request.external_id,
            subject=request.subject,
            body=request.body,
            customer_id=request.customer_id,
            received_at=result.started_at,
        )
        session.add(ticket)
        run = _run_row(ticket.id, result)
        session.add(run)
        _add_trace_rows(session, result)
        session.flush()
        session.expunge(run)
        return run


def _run_row(ticket_id: str, result: RunResult) -> TriageRun:
    resolution = result.resolution
    classification = result.classification
    action = resolution.recommended_action if resolution else None
    return TriageRun(
        id=result.run_id,
        ticket_id=ticket_id,
        status=result.status.value,
        error_kind=result.error_kind.value if result.error_kind else None,
        error_detail=result.error_detail,
        category=classification.category.value if classification else None,
        urgency=classification.urgency.value if classification else None,
        confidence=(
            min(classification.confidence, resolution.confidence)
            if classification and resolution
            else None
        ),
        answer_status=resolution.answer_status if resolution else None,
        reply_draft=resolution.reply_draft if resolution else None,
        recommended_action=action.kind.value if action else None,
        action_executed=bool(result.action_result and result.action_result.ok),
        escalation_reasons=",".join(reason.value for reason in result.escalation_reasons),
        rationale=result.rationale,
        prompt_bundle_version=result.prompt_bundle_version,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        provider_call_count=result.provider_call_count,
        latency_ms=result.latency_ms,
        started_at=result.started_at,
        completed_at=result.completed_at,
    )


def _add_trace_rows(session: Session, result: RunResult) -> None:
    for step in result.steps:
        session.add(
            RunStep(
                run_id=result.run_id,
                seq=step.seq,
                kind=step.kind.value,
                status=step.status.value,
                latency_ms=step.latency_ms,
                note=step.note,
            )
        )
    for seq, call in enumerate(result.calls, 1):
        session.add(
            LLMCall(
                run_id=result.run_id,
                seq=seq,
                step_kind=call.step_kind,
                prompt_name=call.prompt_name,
                prompt_version=call.prompt_version,
                provider=call.provider,
                model=call.model,
                attempt_kind=call.attempt_kind,
                attempt_number=call.attempt_number,
                outcome=call.outcome,
                request_id=call.request_id,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                estimated_cost_usd=call.estimated_cost_usd,
                latency_ms=call.latency_ms,
                error_detail=call.error_detail,
            )
        )
    for item in result.evidence.items:
        chunk = item.chunk
        session.add(
            EvidenceRecord(
                run_id=result.run_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                document_version=chunk.document_version,
                heading=chunk.heading,
                score=item.score,
                rank=item.rank,
                was_cited=chunk.chunk_id in result.cited,
                cited_quote=result.cited.get(chunk.chunk_id),
            )
        )
    for seq, (outcome, attempts) in enumerate(result.tool_attempts, 1):
        session.add(
            ToolCallRecord(
                run_id=result.run_id,
                seq=seq,
                tool_name=outcome.tool_name,
                arguments_json=json.dumps(outcome.arguments, sort_keys=True)
                if outcome.arguments
                else None,
                status=outcome.status,
                attempt_number=attempts,
                result_summary=outcome.rendered,
                error_detail=outcome.error,
                idempotency_key=None,
                latency_ms=0,
            )
        )
    if result.status is TriageStatus.NEEDS_HUMAN_REVIEW:
        session.add(
            Review(
                run_id=result.run_id,
                state="pending",
                reasons=",".join(r.value for r in result.escalation_reasons),
                created_at=result.completed_at,
            )
        )
