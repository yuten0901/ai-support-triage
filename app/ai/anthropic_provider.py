"""The real provider adapter.

The only file in the service that imports a vendor SDK. Everything above it
speaks :class:`LLMRequest` / :class:`LLMResponse`, so switching vendors is a
change here and nowhere else -- and the entire test suite and evaluation run
without the SDK installed at all, because nothing on the default path imports
this module.

Three decisions in here are worth more than the code around them.

**The SDK's own retries are turned off.** ``max_retries=0`` is deliberate. The
SDK will happily retry 429s and 5xx for you, which sounds helpful and quietly
destroys the thing this service is built to demonstrate: with SDK-level retries
on, a run's trace shows one call that took nine seconds, and the retry count,
the per-attempt latencies and the backoff schedule are invisible. Retry policy
belongs to :mod:`app.ai.client`, where it is bounded, recorded and testable.

**A refusal is detected before the content is read.** ``stop_reason ==
"refusal"`` comes back as a perfectly successful HTTP 200 whose ``content`` may
be empty. Code that reaches for ``content[0].text`` first crashes on it and
reports an outage; the refusal is a decision, not a fault, and it is mapped to
its own terminal state.

**The JSON schema is sanitised before it is sent.** Structured-output mode
accepts a subset of JSON Schema -- no numeric bounds, no string lengths, and
``additionalProperties: false`` required on every object. Pydantic emits some
of what it rejects. Stripping those keywords loses nothing, because the same
constraints are enforced a second time when the response is validated against
the Pydantic model; the provider guarantees the shape and we still check the
content.
"""

from __future__ import annotations

from typing import Any

from app.ai.provider import (
    LLMRequest,
    LLMResponse,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    TokenUsage,
)

NAME = "anthropic"

#: JSON Schema keywords structured-output mode does not accept. Removing them
#: is safe here and only here: the Pydantic model re-validates every response,
#: so a value outside these bounds is still rejected -- one layer later.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
        "format",
        "default",
    }
)


def sanitise_schema(schema: Any) -> Any:
    """Recursively strip unsupported keywords and close every object."""
    if isinstance(schema, list):
        return [sanitise_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    cleaned = {
        key: sanitise_schema(value)
        for key, value in schema.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }
    if cleaned.get("type") == "object":
        cleaned["additionalProperties"] = False
        properties = cleaned.get("properties")
        if isinstance(properties, dict):
            # Strict mode requires every declared property to be listed as
            # required. Optional fields in these schemas are already
            # `T | None`, so requiring them means "send null", not "invent a
            # value" -- and an explicit null is exactly what we want back
            # rather than a silently absent key.
            cleaned["required"] = list(properties)
    return cleaned


class AnthropicProvider:
    """Adapter over the Anthropic Messages API."""

    name = NAME

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_timeout_seconds: float = 30.0,
    ) -> None:
        # Imported here rather than at module scope so the package stays an
        # optional dependency: `pip install .` gets a working service, its
        # tests and its evaluation without it.
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=default_timeout_seconds,
            max_retries=0,  # see module docstring
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
                # The API takes system content in its own field; a system-role
                # entry in `messages` means something different and is not what
                # this service builds.
                if message.role != "system"
            ],
            "output_config": {"effort": request.effort},
        }
        if request.json_schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": sanitise_schema(request.json_schema),
            }

        try:
            raw = self._client.with_options(
                timeout=request.timeout_seconds
            ).messages.with_raw_response.create(**kwargs)
            message = raw.parse()
        except Exception as error:  # noqa: BLE001 - re-raised as our taxonomy below
            raise self._translate(error) from error

        request_id = raw.headers.get("request-id")

        # Checked before touching `content`: a refusal can carry an empty
        # content list, and reading it blind would turn a decision into a
        # crash.
        if getattr(message, "stop_reason", None) == "refusal":
            return LLMResponse(
                text="",
                usage=self._usage(message),
                model=getattr(message, "model", request.model),
                request_id=request_id,
                provider_refusal=True,
                extra={"stop_details": self._stop_details(message)},
            )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

        return LLMResponse(
            text=text,
            usage=self._usage(message),
            model=getattr(message, "model", request.model),
            request_id=request_id,
            provider_refusal=False,
            extra={"stop_reason": getattr(message, "stop_reason", None)},
        )

    @staticmethod
    def _stop_details(message: Any) -> dict[str, Any]:
        details = getattr(message, "stop_details", None)
        if details is None:
            return {}
        return {
            "category": getattr(details, "category", None),
            "explanation": getattr(details, "explanation", None),
        }

    @staticmethod
    def _usage(message: Any) -> TokenUsage:
        """Read usage, or record that it was not reported.

        ``reported=False`` propagates as ``None`` all the way to the usage
        summary. Substituting zeros here would be a small lie that compounds
        into a cost dashboard nobody can trust.
        """
        usage = getattr(message, "usage", None)
        if usage is None:
            return TokenUsage(reported=False)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None or output_tokens is None:
            return TokenUsage(reported=False)
        return TokenUsage(
            input_tokens=int(input_tokens), output_tokens=int(output_tokens), reported=True
        )

    def _translate(self, error: Exception) -> Exception:
        """Map an SDK exception onto the caller-action taxonomy.

        Ordered most specific first. The split is by what the caller should do,
        not by status code: a 429 and a 503 are both "wait and try again", and
        a 400 and a 401 are both "stop, a human must fix something".
        """
        sdk = self._sdk
        request_id = getattr(error, "request_id", None)

        if isinstance(error, sdk.APITimeoutError):
            return ProviderTimeout(str(error), request_id=request_id)
        if isinstance(error, sdk.RateLimitError):
            return ProviderRateLimited(
                str(error), retry_after=self._retry_after(error), request_id=request_id
            )
        if isinstance(error, sdk.AuthenticationError | sdk.PermissionDeniedError):
            return ProviderAuthError(str(error), request_id=request_id)
        if isinstance(
            error, sdk.BadRequestError | sdk.NotFoundError | sdk.UnprocessableEntityError
        ):
            return ProviderBadRequest(str(error), request_id=request_id)
        if isinstance(error, sdk.APIStatusError):
            status = getattr(error, "status_code", 0)
            if status >= 500:
                return ProviderUnavailable(str(error), request_id=request_id)
            return ProviderBadRequest(str(error), request_id=request_id)
        if isinstance(error, sdk.APIConnectionError):
            return ProviderUnavailable(str(error), request_id=request_id)

        # Anything else is a bug in this adapter rather than a provider
        # condition, and is surfaced as such instead of being mislabelled as an
        # outage that someone will wait out.
        return error

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        """Honour ``Retry-After`` when the provider sends one."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        raw = headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            # A date-formatted Retry-After. Falling back to our own schedule is
            # better than parsing HTTP dates for a value we clamp anyway.
            return None
