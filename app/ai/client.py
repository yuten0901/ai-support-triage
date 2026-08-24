"""One structured call to the model, with everything that can go wrong.

This is where the project's central claim lives: an LLM is an unreliable
subsystem, and the correct response is not hope but a bounded protocol. Between
"ask the provider" and "return a validated object" there are five distinct
failure modes, and each one is handled differently on purpose:

===========================  ======================================  ===================
Failure                      Response                                Budget it consumes
===========================  ======================================  ===================
Timeout / 429 / 5xx          Retry with jittered backoff             transport attempts
Auth error / bad request     Stop immediately                        --
Not JSON, or invalid schema  Re-ask with the validator's message     repair attempts
Semantically wrong output    Re-ask with a *specific* correction     repair attempts
Provider-level refusal       Stop, and report it as a refusal        --
===========================  ======================================  ===================

**The two budgets are separate and that separation is load-bearing.** A
transport retry means the provider never answered; a repair means it answered
with something unusable. Those are different problems with different owners --
an outage versus a prompt or model regression -- and a single shared counter
lets a flapping network consume the entire budget for fixing malformed JSON,
after which a perfectly ordinary schema error ends the run as an outage. They
are bounded together by a hard per-step ceiling on provider calls, so no
combination of the two can loop.

**Retrying a non-retryable error is the classic mistake.** A 401 retried three
times with backoff turns a loud, instant, actionable failure into a slow one
that looks like an outage. The taxonomy in ``app/ai/provider.py`` exists so
that decision is made once, by type, and not by reading status codes at the
call site.

**A refusal is not an error.** ``ModelRefusal`` is raised, not caught here, and
becomes ``REJECTED_BY_MODEL`` -- a terminal state distinct from ``FAILED``.
Nothing is broken and retrying will not help; the model declined, which is
sometimes the right answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.ai import pricing
from app.ai.prompts.templates import PromptTemplate
from app.ai.provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Message,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.domain.backoff import JitterSource, compute_delay
from app.domain.clock import Clock
from app.domain.states import ErrorKind, StepKind

T = TypeVar("T", bound=BaseModel)

#: Models are told not to use markdown fences. Some do anyway. Unwrapping one
#: is tolerated -- but recorded (see ``CallRecord.note``), because a silent
#: accommodation is how a formatting regression stays invisible for months.
_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.DOTALL)


class StepFailure(Exception):
    """The step could not produce a valid result. A system fault."""

    def __init__(self, kind: ErrorKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class ModelRefusal(Exception):
    """The provider or the model declined to answer. Not a fault."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One provider round trip, as it will be persisted."""

    step_kind: str
    prompt_name: str
    prompt_version: str
    provider: str
    model: str
    attempt_kind: str
    attempt_number: int
    outcome: str
    latency_ms: int
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error_detail: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class RunLimits:
    """The reliability envelope for one run, lifted out of settings."""

    max_transport_attempts: int
    max_repair_attempts: int
    max_provider_calls_per_step: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float
    request_timeout_seconds: float
    run_deadline_seconds: float
    cost_budget_usd: float
    max_output_tokens: int
    effort: str
    use_structured_outputs: bool


@dataclass
class RunBudget:
    """What the run has spent, and whether it may spend more.

    Deliberately not a decorator or a context manager: the checks happen at
    named points (before each provider call) and the accounting is read
    afterwards by the recorder, so a plain object that can be inspected beats
    something clever that cannot.
    """

    limits: RunLimits
    clock: Clock
    started_monotonic: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    spent_usd: float = 0.0
    #: ``False`` once any call's cost could not be determined. The dollar
    #: budget stops being enforceable at that point, and the run says so rather
    #: than enforcing a limit against an assumed zero.
    cost_known: bool = True

    def __post_init__(self) -> None:
        self.started_monotonic = self.clock.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return self.clock.monotonic() - self.started_monotonic

    @property
    def remaining_seconds(self) -> float:
        return self.limits.run_deadline_seconds - self.elapsed_seconds

    @property
    def provider_call_count(self) -> int:
        return len(self.calls)

    def totals(self) -> tuple[int | None, int | None, float | None]:
        """Reported input tokens, output tokens and cost for the whole run.

        Any call with unreported usage makes the corresponding total ``None``:
        a partial sum presented as a total is a wrong number, and the point of
        the usage summary is that its numbers can be trusted.
        """
        if any(call.input_tokens is None for call in self.calls):
            tokens_in: int | None = None
        else:
            tokens_in = sum(call.input_tokens or 0 for call in self.calls)
        if any(call.output_tokens is None for call in self.calls):
            tokens_out: int | None = None
        else:
            tokens_out = sum(call.output_tokens or 0 for call in self.calls)
        cost = self.spent_usd if self.cost_known else None
        return tokens_in, tokens_out, cost

    def check_deadline(self) -> None:
        if self.remaining_seconds <= 0:
            raise StepFailure(
                ErrorKind.DEADLINE_EXCEEDED,
                f"Run exceeded its {self.limits.run_deadline_seconds:.0f}s deadline "
                f"after {self.provider_call_count} provider calls.",
            )

    def check_affordable(self, model: str, prompt_chars: int) -> None:
        """Stop *before* a call that could push the run over its budget."""
        ceiling = pricing.worst_case_cost(
            model,
            estimated_input_tokens=pricing.tokens_from_chars(prompt_chars),
            max_output_tokens=self.limits.max_output_tokens,
        )
        if ceiling is None:
            # Unpriced model: the dollar budget is unenforceable. The call
            # ceiling and the deadline still bound the run.
            return
        if self.spent_usd + ceiling > self.limits.cost_budget_usd:
            raise StepFailure(
                ErrorKind.BUDGET_EXCEEDED,
                f"Run would exceed its ${self.limits.cost_budget_usd:.2f} budget: "
                f"${self.spent_usd:.4f} spent, next call could cost up to ${ceiling:.4f}.",
            )

    def record(self, call: CallRecord) -> None:
        self.calls.append(call)
        if call.estimated_cost_usd is None:
            self.cost_known = False
        else:
            self.spent_usd += call.estimated_cost_usd


@dataclass(frozen=True, slots=True)
class SemanticValidator:
    """A check the schema cannot express.

    Schema validation proves the *shape*; this proves the *content* -- above
    all, that citations resolve to chunks that were actually shown and quote
    them exactly. A failure here is repairable and gets its own correction
    message, because "your JSON is malformed" and "you cited something that
    does not exist" call for completely different fixes.
    """

    check: Callable[[BaseModel], str | None]
    repair_template: str
    error_kind: ErrorKind


_RETRYABLE_ERROR_KINDS: dict[type[ProviderError], ErrorKind] = {
    ProviderTimeout: ErrorKind.PROVIDER_TIMEOUT,
    ProviderRateLimited: ErrorKind.PROVIDER_RATE_LIMITED,
    ProviderUnavailable: ErrorKind.PROVIDER_UNAVAILABLE,
    ProviderAuthError: ErrorKind.PROVIDER_AUTH,
    ProviderBadRequest: ErrorKind.PROVIDER_BAD_REQUEST,
}


def _error_kind(error: ProviderError) -> ErrorKind:
    for error_type, kind in _RETRYABLE_ERROR_KINDS.items():
        if isinstance(error, error_type):
            return kind
    return ErrorKind.INTERNAL


def _extract_json(text: str) -> tuple[Any, str | None]:
    """Parse the model's text as a JSON object.

    Returns the parsed value and an optional note describing any tolerance that
    had to be applied. Strict first: only if that fails is a markdown fence
    stripped, and then the fact is reported so it shows up in the trace.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    fenced = _FENCE.match(stripped)
    if fenced is not None:
        # Deliberately not caught: if the fence contents are also invalid, the
        # original JSONDecodeError semantics apply and the caller reports it as
        # unparseable output.
        return json.loads(fenced.group("body")), "unwrapped a markdown code fence"

    raise json.JSONDecodeError("model output is not JSON", stripped, 0)


class StructuredCaller:
    """Calls the provider until it produces a valid object, or gives up."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        limits: RunLimits,
        clock: Clock,
        jitter: JitterSource,
        repair_template: str,
    ) -> None:
        self._provider = provider
        self._model = model
        self._limits = limits
        self._clock = clock
        self._jitter = jitter
        self._repair_template = repair_template

    def call(
        self,
        *,
        step_kind: StepKind,
        prompt: PromptTemplate,
        system_extra: str = "",
        user_content: str,
        schema: type[T],
        budget: RunBudget,
        semantic_validator: SemanticValidator | None = None,
    ) -> T:
        """Run one workflow step's model call.

        Raises :class:`StepFailure` for a system fault and
        :class:`ModelRefusal` when the model declined. Returns only a fully
        validated object -- there is no partial-success path, by design.
        """
        system = prompt.system + system_extra
        messages: list[Message] = [Message(role="user", content=user_content)]

        transport_attempts = 0
        repair_attempts = 0
        calls_this_step = 0
        # Tracked explicitly rather than derived from the counters: once a
        # repair has happened, a later transport retry would otherwise be
        # mislabelled as a repair, and the two budgets would be impossible to
        # tell apart in the trace -- which is the one thing the trace is for.
        attempt_kind = "initial"

        while True:
            budget.check_deadline()
            if calls_this_step >= self._limits.max_provider_calls_per_step:
                # Reachable only if the two per-kind budgets are configured
                # above this ceiling. It is the backstop that makes the loop
                # provably finite regardless of configuration.
                raise StepFailure(
                    ErrorKind.INVALID_MODEL_OUTPUT,
                    f"{step_kind.value}: hit the hard ceiling of "
                    f"{self._limits.max_provider_calls_per_step} provider calls for one step.",
                )

            prompt_chars = len(system) + sum(len(m.content) for m in messages)
            budget.check_affordable(self._model, prompt_chars)

            request = LLMRequest(
                system=system,
                messages=tuple(messages),
                model=self._model,
                max_output_tokens=self._limits.max_output_tokens,
                schema_name=schema.__name__,
                json_schema=schema.model_json_schema()
                if self._limits.use_structured_outputs
                else None,
                effort=self._limits.effort,
                # Never let one call outlive the run's remaining time.
                timeout_seconds=max(
                    0.1, min(self._limits.request_timeout_seconds, budget.remaining_seconds)
                ),
            )

            started = self._clock.monotonic()
            calls_this_step += 1
            try:
                response = self._provider.complete(request)
            except ProviderError as error:
                latency_ms = int((self._clock.monotonic() - started) * 1000)
                kind = _error_kind(error)
                budget.record(
                    CallRecord(
                        step_kind=step_kind.value,
                        prompt_name=prompt.name,
                        prompt_version=prompt.version,
                        provider=self._provider.name,
                        model=self._model,
                        attempt_kind=attempt_kind,
                        attempt_number=calls_this_step,
                        outcome=type(error).__name__,
                        latency_ms=latency_ms,
                        request_id=error.request_id,
                        error_detail=str(error),
                    )
                )

                if not error.retryable:
                    raise StepFailure(
                        kind, f"{step_kind.value}: {type(error).__name__}: {error}"
                    ) from error

                transport_attempts += 1
                if transport_attempts >= self._limits.max_transport_attempts:
                    raise StepFailure(
                        kind,
                        f"{step_kind.value}: {type(error).__name__} after "
                        f"{transport_attempts} attempts: {error}",
                    ) from error

                retry_after = getattr(error, "retry_after", None)
                delay = compute_delay(
                    transport_attempts,
                    base_seconds=self._limits.retry_base_delay_seconds,
                    max_seconds=self._limits.retry_max_delay_seconds,
                    retry_after=retry_after,
                    jitter=self._jitter,
                )
                self._clock.sleep(delay)
                attempt_kind = "transport_retry"
                continue

            latency_ms = int((self._clock.monotonic() - started) * 1000)
            usage = response.usage
            cost = pricing.estimate_cost(self._model, usage)
            response_for_record = response
            attempt_kind_for_record = attempt_kind
            attempt_number_for_record = calls_this_step
            latency_ms_for_record = latency_ms
            usage_for_record = usage
            cost_for_record = cost

            def _record(
                outcome: str, *, detail: str | None = None, note: str | None = None
            ) -> None:  # noqa: B023 - invoked before the next loop iteration
                budget.record(
                    CallRecord(
                        step_kind=step_kind.value,
                        prompt_name=prompt.name,
                        prompt_version=prompt.version,
                        provider=self._provider.name,
                        model=response_for_record.model or self._model,
                        attempt_kind=attempt_kind_for_record,
                        attempt_number=attempt_number_for_record,
                        outcome=outcome,
                        latency_ms=latency_ms_for_record,
                        request_id=response_for_record.request_id,
                        input_tokens=(
                            usage_for_record.input_tokens if usage_for_record.reported else None
                        ),
                        output_tokens=(
                            usage_for_record.output_tokens if usage_for_record.reported else None
                        ),
                        estimated_cost_usd=cost_for_record,
                        error_detail=detail,
                        note=note,
                    )
                )

            if response.provider_refusal:
                _record("refusal")
                raise ModelRefusal(
                    f"{step_kind.value}: the provider signalled a refusal for this request."
                )

            error_message = self._validate(response, schema)
            if error_message is None:
                parsed, note = _extract_json(response.text)
                value = schema.model_validate(parsed)

                if semantic_validator is not None:
                    semantic_error = semantic_validator.check(value)
                    if semantic_error is not None:
                        _record("semantic_rejected", detail=semantic_error, note=note)
                        repair_attempts += 1
                        if repair_attempts > self._limits.max_repair_attempts:
                            raise StepFailure(
                                semantic_validator.error_kind,
                                f"{step_kind.value}: {semantic_error} "
                                f"(after {repair_attempts - 1} repair attempt(s))",
                            )
                        messages.append(Message(role="assistant", content=response.text))
                        messages.append(
                            Message(
                                role="user",
                                content=semantic_validator.repair_template.format(
                                    error=semantic_error
                                ),
                            )
                        )
                        attempt_kind = "repair"
                        continue

                _record("ok", note=note)
                return value

            _record("invalid_output", detail=error_message)
            repair_attempts += 1
            if repair_attempts > self._limits.max_repair_attempts:
                raise StepFailure(
                    ErrorKind.INVALID_MODEL_OUTPUT,
                    f"{step_kind.value}: {error_message} "
                    f"(after {repair_attempts - 1} repair attempt(s))",
                )
            messages.append(Message(role="assistant", content=response.text))
            messages.append(
                Message(role="user", content=self._repair_template.format(error=error_message))
            )
            attempt_kind = "repair"

    @staticmethod
    def _validate(response: LLMResponse, schema: type[T]) -> str | None:
        """Return a human-readable rejection reason, or ``None`` if valid.

        The message goes straight back to the model, so it is written for a
        reader that has to act on it: which field, what was wrong.
        """
        try:
            parsed, _ = _extract_json(response.text)
        except json.JSONDecodeError as exc:
            preview = response.text.strip()[:200]
            return f"Output was not valid JSON ({exc.msg}). It began: {preview!r}"

        if not isinstance(parsed, dict):
            return f"Output must be a JSON object, got {type(parsed).__name__}."

        try:
            schema.model_validate(parsed)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or '(root)'}: {err['msg']}"
                for err in exc.errors()[:6]
            )
            return f"Output did not match the {schema.__name__} schema: {problems}"
        return None
