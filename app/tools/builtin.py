"""The three read-only tools, and why there are only three.

Every tool is a place the model can steer the system, so the set is kept to
lookups that genuinely change an answer:

* ``lookup_order`` -- the facts a refund or delivery question turns on.
* ``lookup_account`` -- which plan the customer is on, and therefore which of
  two conflicting policy sections applies. Without it, the run detects the
  conflict and escalates instead of guessing.
* ``check_refund_eligibility`` -- date arithmetic against the refund windows.

The third is the interesting one. "Is this order still inside the refund
window?" is a subtraction and a comparison. Asking a language model to do it
would be slower, cost money, and be wrong sometimes -- and being wrong there
means refunding something that should not have been refunded. So the model is
allowed to *ask* the question and never to answer it. That split is the general
rule this service applies everywhere: the model handles language and judgement,
deterministic code handles arithmetic and policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.domain.clock import Clock
from app.tools.registry import Tool, ToolOutcome, ToolRegistry
from app.tools.store import Account, BackingStore, Order

#: Loose enough to accept the real id format, strict enough that a model
#: hallucinating "the customer's order" as an id is rejected before it reaches
#: the store. The pattern is part of the tool's contract and is shown to the
#: model in the catalogue.
_ORDER_ID = r"^ORD-\d{4,8}$"
_ACCOUNT_ID = r"^ACC-\d{4,8}$"

#: Damaged goods are refundable well outside the ordinary window; this bounds
#: "well outside" so the rule is still a rule.
DAMAGED_GOODS_WINDOW_DAYS = 365


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderArgs(_Args):
    order_id: str = Field(
        pattern=_ORDER_ID, description="Order reference exactly as the customer wrote it, e.g. ORD-10042"
    )


class AccountArgs(_Args):
    account_id: str = Field(
        pattern=_ACCOUNT_ID, description="Account reference exactly as the customer wrote it, e.g. ACC-2001"
    )


@dataclass(frozen=True, slots=True)
class RefundRules:
    """Refund windows, in one place.

    Held as data rather than as literals inside the eligibility function so the
    numbers can be stated in a test, and so the two windows are visibly the
    same kind of thing rather than two magic numbers in two branches.
    """

    standard_window_days: int
    enterprise_window_days: int


def _format_date(value: date | None) -> str:
    return value.isoformat() if value else "unknown"


def _order_lines(order: Order) -> str:
    return "\n".join(
        [
            f"order_id: {order.order_id}",
            f"account_id: {order.account_id}",
            f"placed_on: {order.placed_on.isoformat()}",
            f"status: {order.status}",
            f"total_minor: {order.total_minor} {order.currency}",
            f"items: {', '.join(order.items) or 'none recorded'}",
            f"condition: {order.condition}",
            f"final_sale: {str(order.final_sale).lower()}",
            f"estimated_delivery_on: {_format_date(order.estimated_delivery_on)}",
            f"tracking_last_moved_on: {_format_date(order.tracking_last_moved_on)}",
        ]
    )


def build_registry(store: BackingStore, clock: Clock, rules: RefundRules) -> ToolRegistry:
    """Assemble the registry the workflow will use."""

    def lookup_order(args: BaseModel) -> ToolOutcome:
        assert isinstance(args, OrderArgs)
        order = store.get_order(args.order_id)
        if order is None:
            # Not an error. The customer quoted a reference that is not ours,
            # which is a fact worth telling them.
            return ToolOutcome(
                tool_name="lookup_order",
                status="not_found",
                rendered=f"No order exists with id {args.order_id}.",
            )
        return ToolOutcome(
            tool_name="lookup_order",
            status="ok",
            rendered=_order_lines(order),
            data={
                "order_id": order.order_id,
                "account_id": order.account_id,
                "placed_on": order.placed_on.isoformat(),
                "status": order.status,
                "total_minor": order.total_minor,
                "currency": order.currency,
                "condition": order.condition,
                "final_sale": order.final_sale,
            },
        )

    def lookup_account(args: BaseModel) -> ToolOutcome:
        assert isinstance(args, AccountArgs)
        account = store.get_account(args.account_id)
        if account is None:
            return ToolOutcome(
                tool_name="lookup_account",
                status="not_found",
                rendered=f"No account exists with id {args.account_id}.",
            )
        return ToolOutcome(
            tool_name="lookup_account",
            status="ok",
            rendered="\n".join(
                [
                    f"account_id: {account.account_id}",
                    f"plan: {account.plan}",
                    f"opened_on: {account.opened_on.isoformat()}",
                    f"locked: {str(account.locked).lower()}",
                ]
            ),
            data={
                "account_id": account.account_id,
                "plan": account.plan,
                "locked": account.locked,
            },
        )

    def check_refund_eligibility(args: BaseModel) -> ToolOutcome:
        assert isinstance(args, OrderArgs)
        order = store.get_order(args.order_id)
        if order is None:
            return ToolOutcome(
                tool_name="check_refund_eligibility",
                status="not_found",
                rendered=f"No order exists with id {args.order_id}, so eligibility cannot be assessed.",
            )

        account = store.get_account(order.account_id)
        verdict = evaluate_refund_eligibility(
            order=order, account=account, today=clock.now().date(), rules=rules
        )
        return ToolOutcome(
            tool_name="check_refund_eligibility",
            status="ok",
            rendered="\n".join(f"{key}: {value}" for key, value in verdict.items()),
            data=verdict,
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup_order",
            description=(
                "Fetch one order: date placed, status, total, condition and delivery dates. "
                "Use when the answer depends on facts about a specific order."
            ),
            args_model=OrderArgs,
            handler=lookup_order,
        )
    )
    registry.register(
        Tool(
            name="lookup_account",
            description=(
                "Fetch one account: plan tier (standard or enterprise) and lock state. "
                "Use when two policy sections apply to different plans and you need to know which."
            ),
            args_model=AccountArgs,
            handler=lookup_account,
        )
    )
    registry.register(
        Tool(
            name="check_refund_eligibility",
            description=(
                "Deterministic refund-window check for one order: how many days have passed, "
                "which window applies to that account's plan, and the maximum refundable amount. "
                "Use this instead of doing the date arithmetic yourself."
            ),
            args_model=OrderArgs,
            handler=check_refund_eligibility,
        )
    )
    return registry


def evaluate_refund_eligibility(
    *, order: Order, account: Account | None, today: date, rules: RefundRules
) -> dict[str, object]:
    """The refund-window rules, as a pure function.

    Exposed separately from the tool so the rules can be table-tested without
    a registry, a store or a clock. The ordering of the branches is the policy:
    a final-sale item is never refundable, damaged goods override the ordinary
    window, and only then does the plan's window apply.
    """
    days_since_order = (today - order.placed_on).days
    plan = account.plan if account is not None else "unknown"
    window = (
        rules.enterprise_window_days if plan == "enterprise" else rules.standard_window_days
    )

    if order.final_sale:
        eligible, reason = False, "Item was a final-sale purchase."
    elif order.condition in ("damaged", "defective"):
        eligible = days_since_order <= DAMAGED_GOODS_WINDOW_DAYS
        reason = (
            f"Order is recorded as {order.condition}; the damaged-goods window is "
            f"{DAMAGED_GOODS_WINDOW_DAYS} days."
        )
    elif days_since_order <= window:
        eligible, reason = True, f"Within the {window}-day window for the {plan} plan."
    else:
        eligible = False
        reason = (
            f"{days_since_order} days have passed; the {plan} plan window is {window} days."
        )

    return {
        "order_id": order.order_id,
        "plan": plan,
        "days_since_order": days_since_order,
        "window_days": window,
        "condition": order.condition,
        "final_sale": order.final_sale,
        "eligible": eligible,
        "reason": reason,
        "max_refundable_minor": order.total_minor,
        "currency": order.currency,
    }
