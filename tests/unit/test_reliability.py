from datetime import UTC, datetime

import pytest

from app.ai.client import RunBudget, RunLimits, StepFailure, StructuredCaller
from app.ai.prompts import templates
from app.ai.provider import LLMRequest, ProviderTimeout
from app.ai.schemas import Classification
from app.domain.clock import FrozenClock
from app.domain.states import StepKind


class AlwaysTimeout:
    name = "timeout-double"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest):
        self.calls += 1
        raise ProviderTimeout("timed out")


def test_transport_retry_boundary_is_exact() -> None:
    provider = AlwaysTimeout()
    clock = FrozenClock(datetime(2026, 8, 24, tzinfo=UTC))
    limits = RunLimits(
        max_transport_attempts=2,
        max_repair_attempts=0,
        max_provider_calls_per_step=5,
        retry_base_delay_seconds=0.1,
        retry_max_delay_seconds=1,
        request_timeout_seconds=1,
        run_deadline_seconds=10,
        cost_budget_usd=1,
        max_output_tokens=256,
        effort="low",
        use_structured_outputs=True,
    )
    caller = StructuredCaller(
        provider,
        model="claude-opus-5",
        limits=limits,
        clock=clock,
        jitter=lambda: 0,
        repair_template="repair",
    )
    with pytest.raises(StepFailure):
        caller.call(
            step_kind=StepKind.CLASSIFY,
            prompt=templates.CLASSIFY,
            user_content="ticket",
            schema=Classification,
            budget=RunBudget(limits=limits, clock=clock),
        )
    assert provider.calls == 2
