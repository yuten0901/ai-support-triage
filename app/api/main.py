"""FastAPI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.schemas import ReviewDecision, RunSummary, TicketCreate
from app.config import get_settings
from app.db.models import Review, TriageRun
from app.db.session import create_all, create_db_engine, create_session_factory
from app.services import Services, build_services, execute_and_record
from app.workflow.orchestrator import TriageRequest


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_db_engine(settings)
    create_all(engine)
    app.state.services = build_services(settings, create_session_factory(engine))
    yield
    engine.dispose()


app = FastAPI(title="AI Support Triage", version="1.0.0", lifespan=lifespan)


def services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


def authenticate(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = services(request).settings.api_key.get_secret_value()
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


Auth = Annotated[None, Depends(authenticate)]
ServiceDep = Annotated[Services, Depends(services)]


def summarize(run: TriageRun) -> RunSummary:
    return RunSummary(
        run_id=run.id,
        status=run.status,
        category=run.category,
        urgency=run.urgency,
        confidence=run.confidence,
        answer_status=run.answer_status,
        reply_draft=run.reply_draft,
        recommended_action=run.recommended_action,
        action_executed=run.action_executed,
        escalation_reasons=[item for item in run.escalation_reasons.split(",") if item],
        rationale=run.rationale,
        error_kind=run.error_kind,
        provider_call_count=run.provider_call_count,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        estimated_cost_usd=run.estimated_cost_usd,
        latency_ms=run.latency_ms,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


@app.get("/healthz")
def health(svc: ServiceDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": svc.settings.llm_provider,
        "knowledge_chunks": svc.index.chunk_count,
    }


@app.post("/v1/triage", response_model=RunSummary)
def triage(payload: TicketCreate, _auth: Auth, svc: ServiceDep) -> RunSummary:
    if len(payload.body) > svc.settings.max_ticket_body_chars:
        raise HTTPException(status_code=422, detail="Ticket body exceeds configured limit")
    run = execute_and_record(svc, TriageRequest(**payload.model_dump()))
    return summarize(run)


@app.get("/v1/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str, _auth: Auth, svc: ServiceDep) -> RunSummary:
    with svc.sessions() as session:
        run = session.get(TriageRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return summarize(run)


@app.get("/v1/runs/{run_id}/trace")
def get_trace(run_id: str, _auth: Auth, svc: ServiceDep) -> dict[str, Any]:
    query = (
        select(TriageRun)
        .where(TriageRun.id == run_id)
        .options(
            selectinload(TriageRun.steps),
            selectinload(TriageRun.llm_calls),
            selectinload(TriageRun.evidence),
            selectinload(TriageRun.tool_calls),
        )
    )
    with svc.sessions() as session:
        run = session.scalar(query)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": summarize(run).model_dump(mode="json"),
            "steps": [
                {
                    "seq": x.seq,
                    "kind": x.kind,
                    "status": x.status,
                    "latency_ms": x.latency_ms,
                    "note": x.note,
                }
                for x in run.steps
            ],
            "calls": [
                {
                    "seq": x.seq,
                    "step": x.step_kind,
                    "attempt": x.attempt_kind,
                    "outcome": x.outcome,
                    "request_id": x.request_id,
                }
                for x in run.llm_calls
            ],
            "evidence": [
                {
                    "rank": x.rank,
                    "chunk_id": x.chunk_id,
                    "score": x.score,
                    "was_cited": x.was_cited,
                    "quote": x.cited_quote,
                }
                for x in run.evidence
            ],
            "tools": [
                {
                    "seq": x.seq,
                    "name": x.tool_name,
                    "status": x.status,
                    "arguments": x.arguments_json,
                }
                for x in run.tool_calls
            ],
        }


@app.get("/v1/reviews")
def list_reviews(_auth: Auth, svc: ServiceDep) -> list[dict[str, Any]]:
    with svc.sessions() as session:
        rows = session.scalars(select(Review).order_by(Review.created_at)).all()
        return [
            {
                "run_id": row.run_id,
                "state": row.state,
                "reasons": row.reasons,
                "reviewer": row.reviewer,
            }
            for row in rows
        ]


@app.post("/v1/reviews/{run_id}")
def decide_review(
    run_id: str, decision: ReviewDecision, _auth: Auth, svc: ServiceDep
) -> dict[str, str]:
    with svc.sessions.begin() as session:
        review = session.scalar(select(Review).where(Review.run_id == run_id))
        if review is None:
            raise HTTPException(status_code=404, detail="Pending review not found")
        if review.state != "pending":
            raise HTTPException(status_code=409, detail="Review already decided")
        review.state = decision.decision
        review.reviewer = decision.reviewer
        review.decision_note = decision.note
        review.final_reply = decision.final_reply
        review.decided_at = datetime.now(UTC)
        return {"run_id": run_id, "state": review.state}


@app.get("/v1/knowledge")
def knowledge(_auth: Auth, svc: ServiceDep) -> list[dict[str, Any]]:
    return [asdict(summary) for summary in svc.index.documents]
