"""Application configuration.

Every knob is an environment variable so the same image runs unchanged in
development, CI and production. Secrets are typed ``SecretStr`` so they are
redacted from ``repr()`` output and from anything that formats the settings
object (tracebacks, debug endpoints, log records).

The settings that matter most here are not database URLs -- they are the
*reliability* and *cost* boundaries around the model: how many times a step may
call the provider, how much a single triage run is allowed to spend, and above
what refund amount a human has to sign off. Those are business decisions, so
they live in configuration rather than in the code that happens to enforce them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment (or a local ``.env``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Service -----------------------------------------------------------
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="'json' or 'console'")

    # --- Database ----------------------------------------------------------
    # SQLite by default so the whole project runs with `pip install` and nothing
    # else. PostgreSQL is what a deployment would use, and CI runs the entire
    # suite against a real PostgreSQL server as well (see .github/workflows).
    database_url: str = Field(default="sqlite+pysqlite:///./local.sqlite3")
    db_pool_size: int = Field(default=5, ge=1)
    db_pool_max_overflow: int = Field(default=5, ge=0)

    # --- Inbound authentication --------------------------------------------
    api_key: SecretStr = Field(default=SecretStr("dev-triage-api-key"))
    max_ticket_body_chars: int = Field(default=20_000, ge=100)

    # --- LLM provider ------------------------------------------------------
    llm_provider: Literal["fake", "anthropic"] = Field(
        default="fake",
        description=(
            "'fake' is a deterministic in-process stand-in used by tests, the "
            "evaluation suite and the demo. 'anthropic' is the real provider."
        ),
    )
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_base_url: str | None = Field(
        default=None,
        description="Override the API host. Used by tests to point at a local stand-in server.",
    )
    llm_model: str = Field(default="claude-opus-5")
    llm_max_output_tokens: int = Field(default=4096, ge=256)
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(default="low")
    llm_request_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_use_structured_outputs: bool = Field(
        default=True,
        description=(
            "Ask the provider to constrain output to the JSON schema of the step. "
            "This removes most malformed-JSON failures; it does NOT remove the need "
            "for validation, because a schema-valid answer can still cite evidence "
            "that does not exist."
        ),
    )

    # --- Reliability boundaries --------------------------------------------
    # Two independent budgets, deliberately not merged: a transport retry means
    # "the provider did not answer", a repair means "the provider answered with
    # something invalid". Conflating them lets one failure mode consume the
    # other's budget. `llm_max_provider_calls_per_step` is the hard ceiling that
    # bounds both together.
    llm_max_transport_attempts: int = Field(default=3, ge=1)
    llm_max_repair_attempts: int = Field(default=1, ge=0)
    llm_max_provider_calls_per_step: int = Field(default=4, ge=1)
    llm_retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    llm_retry_max_delay_seconds: float = Field(default=8.0, gt=0)

    max_tool_calls_per_run: int = Field(
        default=3,
        ge=0,
        description="Hard step limit. The workflow is a bounded state machine, not an open loop.",
    )
    max_tool_attempts: int = Field(default=2, ge=1, description="Per tool call, for transient failures.")
    run_deadline_seconds: float = Field(default=60.0, gt=0)
    run_cost_budget_usd: float = Field(
        default=0.50,
        gt=0,
        description="A run that would exceed this before a call is aborted as a system error.",
    )

    # --- Retrieval ---------------------------------------------------------
    knowledge_dir: str = Field(default="knowledge")
    retrieval_top_k: int = Field(default=4, ge=1)
    retrieval_min_score: float = Field(
        default=0.15,
        ge=0,
        description="Chunks below this normalised BM25 score are not shown to the model at all.",
    )

    # --- Escalation policy (business rules, not model behaviour) -----------
    auto_refund_cap_minor: int = Field(
        default=5_000,
        ge=0,
        description="Refunds at or above this amount always require human approval (minor units).",
    )
    min_auto_confidence: float = Field(default=0.70, ge=0, le=1)
    refund_window_days: int = Field(default=30, ge=0)

    # `NoDecode` is load-bearing. For a complex field, pydantic-settings parses
    # the environment value as JSON *before* any validator runs, so a plain
    # `billing_refund,shipping_delivery` would raise SettingsError and the
    # validator below would never see it.
    high_risk_categories: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("billing_refund",),
        description="Categories whose write actions always require approval.",
    )

    @field_validator("high_risk_categories", mode="before")
    @classmethod
    def _split_categories(cls, value: object) -> object:
        """Accept ``a,b,c`` from the environment as well as a real sequence."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("log_format")
    @classmethod
    def _check_log_format(cls, value: str) -> str:
        if value not in {"json", "console"}:
            raise ValueError("log_format must be 'json' or 'console'")
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests that patch the environment)."""
    get_settings.cache_clear()
