"""The typed contract between the model and the rest of the system.

Nothing in this service consumes free-form model prose. Each LLM step declares
a Pydantic model here, and the raw text the provider returned is parsed into
one of these or the step fails. The schemas are also what the provider's
structured-output mode is given, so on a real provider the shape is enforced
twice: once by the provider and once by us. That is not redundancy -- the
provider guarantees the *shape*, and only we can check whether the content is
supported by evidence that actually exists.

Two design notes worth reading before changing anything here:

**Tool arguments are a JSON string, not a nested object.** ``ToolPlan`` carries
``arguments_json: str`` rather than ``arguments: dict``. A free-form object
cannot be expressed under strict JSON-schema mode (which requires
``additionalProperties: false`` everywhere), and more importantly each tool has
its *own* argument schema, so validating the arguments belongs to the tool
registry, not to this envelope. The model writes JSON, we parse it, and the
tool's own model validates it -- and a failure at that point is reported back
to the model as a repairable error rather than crashing the run.

**The model may request escalation; it may not grant approval.**
``Resolution.escalation_requested`` is an input to a deterministic policy
decision in :mod:`app.domain.policy`, never the decision itself. There is no
field anywhere in which a model can approve its own high-value action.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.states import ActionKind


class TicketCategory(StrEnum):
    """The routing taxonomy.

    ``OTHER`` exists deliberately. A closed vocabulary with no exit forces the
    model to mislabel anything the taxonomy did not anticipate, and a confident
    wrong label is worse than an honest "I don't have a box for this" -- which
    a human can then act on.
    """

    BILLING_REFUND = "billing_refund"
    SHIPPING_DELIVERY = "shipping_delivery"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_ISSUE = "technical_issue"
    SUBSCRIPTION_CHANGE = "subscription_change"
    GENERAL_QUESTION = "general_question"
    SPAM_OR_NOISE = "spam_or_noise"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class StrictModel(BaseModel):
    """Base for every model-facing schema.

    ``extra="forbid"`` is the point: a provider that invents a field is a
    provider whose output we do not understand, and quietly dropping the field
    would hide a prompt or model change. It also produces
    ``additionalProperties: false`` in the generated JSON schema, which strict
    structured-output mode requires.
    """

    model_config = ConfigDict(extra="forbid")


class ExtractedEntities(StrictModel):
    """Structured facts pulled out of the ticket text.

    Every field is optional and every one is treated as *unverified*: an order
    id extracted from prose is a claim by the model, and the only thing the
    system does with it is hand it to a lookup tool that will say whether it
    exists. Nothing downstream trusts these values on their own.
    """

    order_id: str | None = Field(default=None, description="Order reference mentioned by the customer.")
    account_id: str | None = Field(default=None, description="Account reference mentioned by the customer.")
    amount_minor: int | None = Field(
        default=None,
        description="Any monetary amount the customer named, in minor units (cents).",
    )
    currency: str | None = Field(default=None, description="ISO 4217 code for amount_minor.")
    product: str | None = Field(default=None, description="Product or plan the customer names.")


class Classification(StrictModel):
    """Output of the classify step."""

    category: TicketCategory
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    summary: str = Field(description="One sentence, for the operator queue.")
    #: Free-text retrieval query the model thinks would find the right policy.
    #: Used *in addition to* the ticket text, never instead of it -- see
    #: app/rag/index.py for why both are searched.
    retrieval_query: str


class ToolPlan(StrictModel):
    """Output of the plan-tools step.

    ``tool_name`` is validated against the registry by the caller; a name that
    is not registered aborts the run rather than being guessed at. Inventing a
    plausible tool name is a classic model failure and there is no safe
    interpretation of it.
    """

    needs_tool: bool
    tool_name: str | None = Field(default=None)
    arguments_json: str | None = Field(
        default=None,
        description="JSON object of arguments for tool_name. Validated against the tool's own schema.",
    )
    reason: str = Field(description="Why this tool, or why none is needed.")


class Citation(StrictModel):
    """A claim's evidence.

    ``quote`` is what makes this more than a label. The validator requires it to
    appear verbatim (whitespace-normalised) inside the chunk named by
    ``chunk_id``, so a model cannot cite a real document and then paraphrase it
    into saying something it does not say. Fabricating a citation is therefore
    not a matter of degree -- it is a check that either passes or fails.
    """

    chunk_id: str = Field(description="Exact id of a chunk that was shown to the model.")
    quote: str = Field(description="Verbatim span copied from that chunk.")


class RecommendedAction(StrictModel):
    """What the system should do, beyond replying.

    The model chooses from a closed set; it cannot describe an action the tool
    registry has no implementation for.
    """

    kind: ActionKind
    amount_minor: int | None = Field(default=None, description="Required when kind is issue_refund.")
    target_id: str | None = Field(default=None, description="Order id or ticket ref the action applies to.")
    justification: str


class Resolution(StrictModel):
    """Output of the resolve step: the actual decision."""

    #: ``answered`` requires at least one citation. ``insufficient_evidence`` is
    #: the correct, expected answer when retrieval came back with nothing that
    #: covers the question, and is *not* penalised anywhere in the system.
    answer_status: Literal["answered", "insufficient_evidence", "refused"]
    reply_draft: str = Field(description="Customer-facing draft. Empty when not answered.")
    citations: list[Citation] = Field(default_factory=list)
    recommended_action: RecommendedAction
    confidence: float = Field(ge=0.0, le=1.0)
    risk: Literal["low", "medium", "high"]
    escalation_requested: bool = Field(
        description="The model asking for a human. Advisory: policy decides, not this field."
    )
    escalation_reason: str | None = Field(default=None)
