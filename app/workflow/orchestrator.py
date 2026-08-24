"""The workflow: a bounded state machine, not an agent loop.

```
                 ticket
                   |
              [ precheck ]  deterministic -- can exit here with no model call
                   |
              [ classify ]  LLM -> Classification
                   |
              [ retrieve ]  deterministic BM25 over the knowledge base
                   |
        +--> [ plan tools ]  LLM -> ToolPlan            (at most N iterations)
        |          |
        +----[ execute tool ]  deterministic, validated arguments
                   |
              [ resolve ]   LLM -> Resolution, then citations verified
                   |
              [ decide ]    deterministic policy gate
                   |
           [ execute action ]  only for an approved write action
```

Every arrow is explicit and every loop has a counter. There is no "keep going
until the model says it is finished", because that is a system whose cost and
latency are set by the model's mood. The tool stage repeats at most
``max_tool_calls_per_run`` times; each model call is bounded by its own retry
and repair budgets; the whole run is bounded by a wall-clock deadline and a
dollar budget. Those four limits together mean the worst case is a number you
can write down before deploying it.

Two properties are worth pointing at specifically.

**The model never ends the run in a success state on its own.** It proposes;
:func:`app.domain.policy.decide_outcome` disposes. The gap between "the model
recommended a $400 refund" and "$400 left the company" is a pure function of
the proposal, the evidence and the configured thresholds -- and there is no
field in any schema the model can set to skip it.

**Failure states stay distinct all the way out.** A provider outage, a model
refusal, an empty knowledge base and a decision routed to a human are four
different results here, four different results in the database, and four
different results in the API response. Collapsing any pair of them would make
the service easier to write and impossible to operate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.ai.client import (
    CallRecord,
    ModelRefusal,
    RunBudget,
    RunLimits,
    SemanticValidator,
    StepFailure,
    StructuredCaller,
)
from app.ai.prompts import templates
from app.ai.prompts.render import (
    render_evidence,
    render_ticket,
    render_tool_catalogue,
    render_tool_results,
)
from app.ai.provider import LLMProvider
from app.ai.schemas import Classification, Resolution, ToolPlan
from app.domain.backoff import JitterSource, compute_delay
from app.domain.clock import Clock
from app.domain.evidence import EvidenceSet
from app.domain.policy import (
    InjectionScan,
    Outcome,
    OutcomePolicy,
    decide_outcome,
    precheck,
    scan_for_injection,
)
from app.domain.states import (
    ActionKind,
    ErrorKind,
    EscalationReason,
    StepKind,
    StepStatus,
    TriageStatus,
)
from app.rag.index import KnowledgeIndex
from app.tools.actions import ActionExecutionError, ActionExecutor, ActionResult, idempotency_key
from app.tools.registry import ToolOutcome, ToolPermanentError, ToolRegistry, ToolTransientError
from app.workflow.grounding import mark_cited, validate_resolution


@dataclass(frozen=True, slots=True)
class TriageRequest:
    """One ticket to triage."""

    external_id: str
    subject: str
    body: str
    customer_id: str | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
    seq: int
    kind: StepKind
    status: StepStatus
    latency_ms: int
    note: str | None = None


@dataclass
class RunResult:
    """Everything one run produced. Persisted verbatim by the recorder."""

    run_id: str
    status: TriageStatus
    started_at: datetime
    completed_at: datetime
    latency_ms: int
    prompt_bundle_version: str
    provider: str
    model: str

    steps: list[StepRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = field(default_factory=list)
    tool_attempts: list[tuple[ToolOutcome, int]] = field(default_factory=list)
    evidence: EvidenceSet = field(default_factory=EvidenceSet)
    cited: dict[str, str] = field(default_factory=dict)

    classification: Classification | None = None
    resolution: Resolution | None = None
    outcome: Outcome | None = None
    injection: InjectionScan = field(default_factory=lambda: InjectionScan(suspected=False))

    error_kind: ErrorKind | None = None
    error_detail: str | None = None
    rationale: str = ""
    action_result: ActionResult | None = None
    action_key: str | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    provider_call_count: int = 0
    #: ``False`` when at least one call's cost could not be determined, so the
    #: dollar budget could not be enforced for the whole run. Surfaced rather
    #: than hidden: an unenforceable budget is an operational fact.
    cost_budget_enforceable: bool = True

    @property
    def escalation_reasons(self) -> list[EscalationReason]:
        return list(self.outcome.escalation_reasons) if self.outcome else []


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Everything the workflow needs from settings, resolved once."""

    limits: RunLimits
    policy: OutcomePolicy
    max_tool_calls_per_run: int
    max_tool_attempts: int
    retrieval_top_k: int
    retrieval_min_score: float
    max_ticket_body_chars: int
    model: str


class Orchestrator:
    """Runs one ticket through the workflow."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        index: KnowledgeIndex,
        tools: ToolRegistry,
        actions: ActionExecutor,
        clock: Clock,
        jitter: JitterSource,
        config: OrchestratorConfig,
    ) -> None:
        self._provider = provider
        self._index = index
        self._tools = tools
        self._actions = actions
        self._clock = clock
        self._jitter = jitter
        self._config = config

    def run(self, request: TriageRequest) -> RunResult:
        started_at = self._clock.now()
        started_monotonic = self._clock.monotonic()
        budget = RunBudget(limits=self._config.limits, clock=self._clock)
        caller = StructuredCaller(
            self._provider,
            model=self._config.model,
            limits=self._config.limits,
            clock=self._clock,
            jitter=self._jitter,
            repair_template=templates.REPAIR_INSTRUCTION,
        )

        result = RunResult(
            run_id=uuid.uuid4().hex,
            status=TriageStatus.FAILED,
            started_at=started_at,
            completed_at=started_at,
            latency_ms=0,
            prompt_bundle_version=templates.bundle_version(),
            provider=self._provider.name,
            model=self._config.model,
        )

        try:
            self._execute(request, result, budget, caller)
        except StepFailure as failure:
            result.status = TriageStatus.FAILED
            result.error_kind = failure.kind
            result.error_detail = failure.detail
            result.rationale = failure.detail
            self._add_step(result, StepKind.DECIDE, StepStatus.FAILED, 0, failure.detail)
        except ModelRefusal as refusal:
            # Not a fault. The model declined; retrying will not change that,
            # and reporting it as an outage would send someone to look at
            # infrastructure that is working perfectly.
            result.status = TriageStatus.REJECTED_BY_MODEL
            result.rationale = refusal.detail
            self._add_step(result, StepKind.DECIDE, StepStatus.DECLINED, 0, refusal.detail)

        result.calls = list(budget.calls)
        result.provider_call_count = budget.provider_call_count
        result.input_tokens, result.output_tokens, result.estimated_cost_usd = budget.totals()
        result.cost_budget_enforceable = budget.cost_known
        result.completed_at = self._clock.now()
        result.latency_ms = int((self._clock.monotonic() - started_monotonic) * 1000)
        return result

    # -- the state machine -------------------------------------------------

    def _execute(
        self,
        request: TriageRequest,
        result: RunResult,
        budget: RunBudget,
        caller: StructuredCaller,
    ) -> None:
        # --- precheck: the cheapest possible answer ------------------------
        started = self._clock.monotonic()
        decision = precheck(request.subject, request.body)
        result.injection = scan_for_injection(request.subject, request.body)
        note = decision.note or (
            f"Injection markers: {', '.join(result.injection.patterns)}"
            if result.injection.suspected
            else "No deterministic shortcut applied."
        )
        self._add_step(
            result,
            StepKind.PRECHECK,
            StepStatus.OK if decision.handled else StepStatus.SKIPPED,
            self._elapsed_ms(started),
            note,
        )
        if decision.handled:
            # A valid "nothing to do" -- and zero provider calls to reach it.
            result.status = TriageStatus.NO_ACTION_REQUIRED
            result.rationale = decision.note or "Handled by the deterministic pre-check."
            return

        ticket_block = render_ticket(
            request.subject, request.body, max_chars=self._config.max_ticket_body_chars
        )

        # --- classify ------------------------------------------------------
        started = self._clock.monotonic()
        classification = caller.call(
            step_kind=StepKind.CLASSIFY,
            prompt=templates.CLASSIFY,
            user_content=ticket_block,
            schema=Classification,
            budget=budget,
        )
        result.classification = classification
        self._add_step(
            result,
            StepKind.CLASSIFY,
            StepStatus.OK,
            self._elapsed_ms(started),
            f"{classification.category.value} / {classification.urgency.value} "
            f"@ confidence {classification.confidence:.2f}",
        )

        # --- retrieve ------------------------------------------------------
        started = self._clock.monotonic()
        # Both the raw ticket and the model's own query. The ticket carries the
        # customer's wording, the query carries the vocabulary the policy uses,
        # and either alone misses cases the other finds.
        query = f"{request.subject} {request.body} {classification.retrieval_query}"
        evidence = self._index.search(
            query,
            top_k=self._config.retrieval_top_k,
            min_score=self._config.retrieval_min_score,
        )
        result.evidence = evidence
        self._add_step(
            result,
            StepKind.RETRIEVE,
            StepStatus.OK if evidence else StepStatus.DECLINED,
            self._elapsed_ms(started),
            f"{len(evidence)} chunk(s) above score {self._config.retrieval_min_score}"
            + (f"; audiences {sorted(evidence.audiences)}" if evidence.audiences else ""),
        )

        # --- tools ---------------------------------------------------------
        self._run_tool_stage(result, budget, caller, ticket_block)

        # --- resolve -------------------------------------------------------
        started = self._clock.monotonic()
        resolution = caller.call(
            step_kind=StepKind.RESOLVE,
            prompt=templates.RESOLVE,
            user_content="\n\n".join(
                [
                    ticket_block,
                    render_evidence(evidence),
                    render_tool_results(result.tool_outcomes),
                ]
            ),
            schema=Resolution,
            budget=budget,
            semantic_validator=SemanticValidator(
                check=lambda value: validate_resolution(
                    value if isinstance(value, Resolution) else Resolution.model_validate(value),
                    evidence,
                ),
                repair_template=templates.UNGROUNDED_CITATION_INSTRUCTION,
                error_kind=ErrorKind.UNGROUNDED_CITATION,
            ),
        )
        result.resolution = resolution
        result.cited = mark_cited(resolution, evidence)
        self._add_step(
            result,
            StepKind.RESOLVE,
            StepStatus.OK if resolution.answer_status == "answered" else StepStatus.DECLINED,
            self._elapsed_ms(started),
            f"{resolution.answer_status}; {len(resolution.citations)} citation(s) verified",
        )

        # --- decide --------------------------------------------------------
        started = self._clock.monotonic()
        outcome = decide_outcome(
            resolution,
            category=classification.category.value,
            model_confidence=classification.confidence,
            evidence=evidence,
            tool_failure_count=self._tool_failure_count(result),
            injection=result.injection,
            account_plan_known=self._account_plan_known(result),
            policy=self._config.policy,
        )
        result.outcome = outcome
        result.status = outcome.status
        result.rationale = outcome.rationale
        self._add_step(
            result,
            StepKind.DECIDE,
            StepStatus.OK,
            self._elapsed_ms(started),
            outcome.rationale,
        )

        # --- execute --------------------------------------------------------
        self._run_action_stage(result)

    # -- tool stage ---------------------------------------------------------

    def _run_tool_stage(
        self,
        result: RunResult,
        budget: RunBudget,
        caller: StructuredCaller,
        ticket_block: str,
    ) -> None:
        if self._config.max_tool_calls_per_run == 0:
            self._add_step(
                result, StepKind.PLAN_TOOLS, StepStatus.SKIPPED, 0, "Tool use is disabled."
            )
            self._add_step(result, StepKind.EXECUTE_TOOL, StepStatus.SKIPPED, 0, None)
            return

        catalogue = render_tool_catalogue(self._tools.catalogue())
        executed = 0

        for _iteration in range(self._config.max_tool_calls_per_run):
            started = self._clock.monotonic()
            plan = caller.call(
                step_kind=StepKind.PLAN_TOOLS,
                prompt=templates.PLAN_TOOLS,
                system_extra="",
                user_content="\n\n".join(
                    [
                        templates.PLAN_TOOLS.system.split("<tools>")[0][:0] or "",
                        ticket_block,
                        f"<tools>\n{catalogue}\n</tools>",
                        render_tool_results(result.tool_outcomes),
                    ]
                ).strip(),
                schema=ToolPlan,
                budget=budget,
            )
            self._add_step(
                result,
                StepKind.PLAN_TOOLS,
                StepStatus.OK if plan.needs_tool else StepStatus.DECLINED,
                self._elapsed_ms(started),
                plan.reason,
            )

            if not plan.needs_tool:
                break

            if not plan.tool_name:
                # Schema-valid and semantically incoherent: "I need a tool" with
                # no tool named. Fed back as a normal correctable result rather
                # than crashing -- and it consumes an iteration, so this cannot
                # become a loop.
                outcome = ToolOutcome(
                    tool_name="(unnamed)",
                    status="invalid_arguments",
                    rendered="needs_tool was true but no tool_name was given.",
                    error="needs_tool without tool_name",
                )
                result.tool_outcomes.append(outcome)
                result.tool_attempts.append((outcome, 1))
                self._add_step(result, StepKind.EXECUTE_TOOL, StepStatus.FAILED, 0, outcome.error)
                continue

            started = self._clock.monotonic()
            outcome, attempts = self._invoke_tool(plan.tool_name, plan.arguments_json)
            executed += 1
            result.tool_outcomes.append(outcome)
            result.tool_attempts.append((outcome, attempts))
            self._add_step(
                result,
                StepKind.EXECUTE_TOOL,
                StepStatus.OK if outcome.succeeded else StepStatus.FAILED,
                self._elapsed_ms(started),
                f"{outcome.tool_name}: {outcome.status}",
            )

        if executed == 0 and not result.tool_outcomes:
            self._add_step(
                result, StepKind.EXECUTE_TOOL, StepStatus.SKIPPED, 0, "No lookup was needed."
            )

    def _invoke_tool(self, name: str, arguments_json: str | None) -> tuple[ToolOutcome, int]:
        """Run one tool, retrying only genuinely transient failures.

        A permanent failure is not retried: repeating a guaranteed failure
        makes an outage slower to diagnose without making it less of an
        outage. Either exhaustion ends the run -- answering as though the
        lookup had returned nothing would turn an unknown into a fabricated
        fact, which is exactly the failure this service exists to prevent.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_tool_attempts + 1):
            try:
                return self._tools.invoke(name, arguments_json), attempt
            except ToolTransientError as error:
                last_error = error
                if attempt < self._config.max_tool_attempts:
                    self._clock.sleep(
                        compute_delay(
                            attempt,
                            base_seconds=self._config.limits.retry_base_delay_seconds,
                            max_seconds=self._config.limits.retry_max_delay_seconds,
                            jitter=self._jitter,
                        )
                    )
            except ToolPermanentError as error:
                raise StepFailure(ErrorKind.TOOL_INFRASTRUCTURE, f"tool {name}: {error}") from error

        raise StepFailure(
            ErrorKind.TOOL_INFRASTRUCTURE,
            f"tool {name} failed after {self._config.max_tool_attempts} attempts: {last_error}",
        )

    @staticmethod
    def _tool_failure_count(result: RunResult) -> int:
        """Failures only. ``not_found`` is an answer, and does not count."""
        return sum(1 for outcome in result.tool_outcomes if not outcome.succeeded)

    @staticmethod
    def _account_plan_known(result: RunResult) -> bool:
        """Whether a lookup established which plan this customer is on.

        Feeds the conflicting-evidence gate. Without a known plan, two policy
        sections addressed to different audiences are a genuine ambiguity and
        the run goes to a human instead of picking one.
        """
        for outcome in result.tool_outcomes:
            data = outcome.data or {}
            if outcome.tool_name == "lookup_account" and outcome.status == "ok":
                return True
            if outcome.tool_name == "check_refund_eligibility" and data.get("plan") not in (
                None,
                "unknown",
            ):
                return True
        return False

    # -- action stage -------------------------------------------------------

    def _run_action_stage(self, result: RunResult) -> None:
        outcome = result.outcome
        assert outcome is not None

        if result.status is not TriageStatus.AUTO_RESOLVED or outcome.action is None:
            reason = (
                "Held for human approval; nothing executed."
                if result.status is TriageStatus.NEEDS_HUMAN_REVIEW
                else "No write action to execute."
            )
            self._add_step(result, StepKind.EXECUTE_ACTION, StepStatus.SKIPPED, 0, reason)
            return

        resolution = result.resolution
        assert resolution is not None
        action = resolution.recommended_action
        target = action.target_id or ""
        key = idempotency_key(result.run_id, outcome.action.value, target, action.amount_minor)
        result.action_key = key

        started = self._clock.monotonic()
        try:
            if outcome.action is ActionKind.ISSUE_REFUND:
                executed = self._actions.issue_refund(
                    order_id=target,
                    amount_minor=action.amount_minor or 0,
                    currency=(
                        result.classification.entities.currency if result.classification else None
                    )
                    or "USD",
                    key=key,
                )
            else:
                executed = self._actions.create_escalation(
                    run_id=result.run_id,
                    reason=outcome.action.value,
                    summary=action.justification,
                    key=key,
                )
        except ActionExecutionError as error:
            # The decision was sound and the execution failed. That is a system
            # fault, and it must not be reported as a resolved ticket -- the
            # customer was told something would happen and it did not.
            result.status = TriageStatus.FAILED
            result.error_kind = ErrorKind.TOOL_INFRASTRUCTURE
            result.error_detail = f"action {outcome.action.value} failed: {error}"
            result.rationale = result.error_detail
            self._add_step(
                result,
                StepKind.EXECUTE_ACTION,
                StepStatus.FAILED,
                self._elapsed_ms(started),
                result.error_detail,
            )
            return

        result.action_result = executed
        self._add_step(
            result,
            StepKind.EXECUTE_ACTION,
            StepStatus.OK,
            self._elapsed_ms(started),
            f"{outcome.action.value}: {executed.status} {executed.reference or ''}".strip(),
        )

    # -- helpers ------------------------------------------------------------

    def _elapsed_ms(self, started: float) -> int:
        return int((self._clock.monotonic() - started) * 1000)

    @staticmethod
    def _add_step(
        result: RunResult,
        kind: StepKind,
        status: StepStatus,
        latency_ms: int,
        note: str | None,
    ) -> None:
        result.steps.append(
            StepRecord(
                seq=len(result.steps) + 1,
                kind=kind,
                status=status,
                latency_ms=latency_ms,
                note=note,
            )
        )
