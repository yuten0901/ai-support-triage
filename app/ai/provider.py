"""The provider boundary.

Everything above this line speaks in :class:`LLMRequest` / :class:`LLMResponse`
and the five error types below. Nothing else in the service imports a vendor
SDK, which is what makes "swap the provider" a one-file change and what lets
the entire test suite and evaluation run without an API key.

The error taxonomy is the important part. It is small on purpose, and the split
is by **what the caller should do**, not by HTTP status:

===========================  ================================================
Error                        Caller's correct response
===========================  ================================================
``ProviderTimeout``          Retry. The request may or may not have run.
``ProviderRateLimited``      Retry, after ``retry_after`` if the provider said.
``ProviderUnavailable``      Retry. The provider is down or overloaded.
``ProviderAuthError``        Do not retry. A human must fix configuration.
``ProviderBadRequest``       Do not retry. Our request is wrong.
===========================  ================================================

Retrying a ``ProviderBadRequest`` is the classic mistake: it turns a fast,
loud, actionable failure into a slow one that looks like an outage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One provider call.

    ``system`` is kept separate from ``messages`` rather than being messages[0].
    That separation is the trust boundary: system content is authored by us,
    everything in ``messages`` contains text we did not write (ticket bodies,
    retrieved documents, tool results). See docs/security.md.
    """

    system: str
    messages: tuple[Message, ...]
    model: str
    max_output_tokens: int
    #: Name of the structured-output schema this call must satisfy, for logging.
    schema_name: str
    #: JSON schema the provider should constrain output to, when it supports it.
    #: ``None`` means "ask in the prompt and validate afterwards" -- which we do
    #: in both cases anyway.
    json_schema: dict[str, Any] | None = None
    effort: str = "low"
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """What the call cost, in tokens. Zero is a valid, meaningful value.

    A provider that does not report usage yields zeros, and the cost estimate
    that follows is then also zero. That is recorded as-is rather than guessed
    at: an invented token count would quietly corrupt the usage summary, and
    "we don't know" is a fact an operator can act on.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    #: Set when the provider tells us; ``None`` means it did not report usage.
    reported: bool = True


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: TokenUsage
    model: str
    #: Provider-side request id, when available. The single most useful field
    #: to have in a log line when asking a vendor what happened.
    request_id: str | None = None
    #: ``True`` when the provider itself signalled a refusal (rather than the
    #: model writing a refusal into its JSON output).
    provider_refusal: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class. Never raised directly."""

    retryable: bool = False

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class ProviderTimeout(ProviderError):
    retryable = True


class ProviderRateLimited(ProviderError):
    retryable = True

    def __init__(
        self, message: str, *, retry_after: float | None = None, request_id: str | None = None
    ) -> None:
        super().__init__(message, request_id=request_id)
        #: Seconds the provider asked us to wait. Honoured, but clamped -- see
        #: app/domain/backoff.py.
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    retryable = True


class ProviderAuthError(ProviderError):
    """Bad or missing credentials. Retrying cannot fix this."""

    retryable = False


class ProviderBadRequest(ProviderError):
    """We sent something the provider rejected. Our bug, not an outage."""

    retryable = False


class LLMProvider(Protocol):
    """The whole provider surface. One method."""

    #: Identifies the provider in logs and in the usage summary.
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion, or raise one of the errors above."""
        ...
