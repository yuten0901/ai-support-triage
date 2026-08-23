"""The tool boundary.

The model never executes anything. It emits a tool *name* and a JSON *string*,
and this module decides whether that pair corresponds to something real and
safe. Three checks, in order, and each one has a distinct failure that is
reported distinctly:

1. **Is the name registered?** An unregistered name is not guessed at, fuzzy
   matched, or ignored. Inventing a plausible tool name is a classic model
   failure and there is no safe interpretation of it.
2. **Is the argument string JSON?** The model writes a string; strings are not
   always JSON.
3. **Do the arguments satisfy the tool's own schema?** Each tool owns its
   parameter model with ``extra="forbid"``, so an unexpected field is a
   rejection rather than a silently dropped argument.

All three produce a *repairable* outcome that is fed back to the model, not an
exception that ends the run. A model that mistyped an argument can fix it; a
crash cannot be fixed by anyone at 3am.

**The tools here are read-only.** Nothing the model can request moves money or
changes an account. Write actions exist (``app/tools/actions.py``) but they are
reachable only from the deterministic policy layer, after the outcome gate has
decided -- so "the model asked for a refund" and "a refund happened" are
separated by a rule the model cannot address.

The last distinction is the one most systems get wrong: **"not found" is a
successful call.** An order id that does not exist is an *answer* -- and often
the most important one, since it means the customer quoted a reference that
isn't ours. Only a broken backing store is a failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

ToolStatus = Literal["ok", "not_found", "invalid_arguments", "failed"]


class ToolTransientError(Exception):
    """The tool could not run, but running it again might work.

    A dropped connection to the order service. Retried within
    ``max_tool_attempts``; if the retries are exhausted the run ends as a
    system error, because the alternative -- answering as though the lookup had
    returned nothing -- is a fabricated fact.
    """


class ToolPermanentError(Exception):
    """The tool cannot run and retrying will not change that.

    A misconfigured data path, a corrupt store. Not retried: repeating a
    guaranteed failure only makes the outage slower to diagnose.
    """


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one tool invocation, as recorded and as shown to the model."""

    tool_name: str
    status: ToolStatus
    #: What the model sees. Plain text, because a nested JSON blob inside a
    #: prompt is harder for a model to read accurately than a short list of
    #: labelled facts -- and harder for a human reading the trace, too.
    rendered: str
    #: Structured result, for the deterministic layers. ``None`` unless ``ok``.
    data: dict[str, Any] | None = None
    #: The arguments that were actually used, re-serialised from the validated
    #: model. Never the raw string -- that may not be JSON at all.
    arguments: dict[str, Any] | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the call produced an answer. ``not_found`` counts."""
        return self.status in ("ok", "not_found")


@dataclass(frozen=True, slots=True)
class Tool:
    """One registered read-only lookup."""

    name: str
    description: str
    #: Pydantic model for the arguments. Owns its own validation; the envelope
    #: schema in ``app/ai/schemas.py`` deliberately knows nothing about it.
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], ToolOutcome]

    def schema_hint(self) -> str:
        """Compact argument description for the prompt catalogue."""
        properties = self.args_model.model_json_schema().get("properties", {})
        required = set(self.args_model.model_json_schema().get("required", []))
        parts = [
            f"{name} ({spec.get('type', 'any')}{'' if name in required else ', optional'})"
            f"{' - ' + spec['description'] if spec.get('description') else ''}"
            for name, spec in properties.items()
        ]
        return "; ".join(parts) or "none"


@dataclass
class ToolRegistry:
    """The complete set of tools the model may request."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalogue(self) -> list[tuple[str, str, str]]:
        """``(name, description, argument hint)`` for each tool, name-sorted.

        Sorted so the rendered prompt is byte-identical between processes.
        A prompt that varies with dictionary ordering makes runs
        non-reproducible and defeats provider-side prompt caching.
        """
        return [
            (tool.name, tool.description, tool.schema_hint())
            for tool in (self._tools[name] for name in self.names())
        ]

    def invoke(self, name: str, arguments_json: str | None) -> ToolOutcome:
        """Validate and run one tool request.

        Raises only :class:`ToolTransientError` / :class:`ToolPermanentError`,
        which mean the *infrastructure* failed. Anything attributable to the
        model comes back as a ``ToolOutcome`` it can be shown and can correct.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(
                tool_name=name,
                status="invalid_arguments",
                rendered=(
                    f"No tool named {name!r} exists. "
                    f"Available tools: {', '.join(self.names()) or 'none'}."
                ),
                error=f"unknown tool: {name}",
            )

        raw = arguments_json if arguments_json is not None else "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolOutcome(
                tool_name=name,
                status="invalid_arguments",
                rendered=f"The arguments were not valid JSON: {exc}. Send a JSON object.",
                error=f"arguments_json is not valid JSON: {exc}",
            )

        if not isinstance(parsed, dict):
            return ToolOutcome(
                tool_name=name,
                status="invalid_arguments",
                rendered="The arguments must be a JSON object, not a list or a scalar.",
                error="arguments_json is not a JSON object",
            )

        try:
            args = tool.args_model.model_validate(parsed)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
                for err in exc.errors()
            )
            return ToolOutcome(
                tool_name=name,
                status="invalid_arguments",
                rendered=f"The arguments were rejected: {detail}",
                error=detail,
            )

        outcome = tool.handler(args)
        # The handler builds the outcome, but the arguments recorded are always
        # the validated ones -- one place, so a handler cannot record something
        # other than what it was given.
        return ToolOutcome(
            tool_name=outcome.tool_name,
            status=outcome.status,
            rendered=outcome.rendered,
            data=outcome.data,
            arguments=args.model_dump(mode="json"),
            error=outcome.error,
        )
