"""The systems the tools read from.

A real deployment would call an order service and a billing API. Here those are
two JSON files, behind a protocol so the tools depend on the *interface* rather
than on the files -- which is what lets a test inject a store that times out
without patching anything and without the tools knowing.

The protocol makes the distinction the whole design rests on explicit: a lookup
that finds nothing returns ``None``, and a store that cannot answer raises. The
two are different facts and are never collapsed into one.

**Dates in the fixtures are relative, not absolute.** An order is recorded as
``days_ago: 12``, resolved against the injected clock at load time. Absolute
dates would mean the demo quietly rots -- the "inside the refund window" order
becomes an outside-the-window order some weeks after this is written, and the
first person to clone the repository sees a different answer than the README
promises. Relative offsets keep the fixture meaning what it says, forever,
while staying perfectly deterministic for a test that pins the clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

from app.domain.clock import Clock
from app.tools.registry import ToolPermanentError


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    account_id: str
    placed_on: date
    total_minor: int
    currency: str
    status: str
    items: tuple[str, ...]
    #: ``ok``, ``damaged`` or ``defective``. Drives the damaged-goods policy
    #: path, which has a far longer window -- so this field decides whether the
    #: ordinary 30-day rule is even the relevant one.
    condition: str
    final_sale: bool
    estimated_delivery_on: date | None
    tracking_last_moved_on: date | None


@dataclass(frozen=True, slots=True)
class Account:
    account_id: str
    #: ``standard`` or ``enterprise``. Selects between two policy sections that
    #: give different answers; without it the run has conflicting evidence and
    #: goes to a human rather than picking one.
    plan: str
    opened_on: date
    locked: bool


class BackingStore(Protocol):
    """What the read-only tools need.

    ``None`` means "no such record" -- a real answer. An exception means the
    store failed and the answer is unknown.
    """

    def get_order(self, order_id: str) -> Order | None: ...

    def get_account(self, account_id: str) -> Account | None: ...


def _offset(row: dict[str, Any], field: str, today: date, where: str) -> date:
    value = row.get(field)
    if not isinstance(value, int):
        raise ToolPermanentError(f"{where}: {field} must be an integer number of days")
    return today - timedelta(days=value)


def _optional_offset(row: dict[str, Any], field: str, today: date, where: str) -> date | None:
    if row.get(field) is None:
        return None
    return _offset(row, field, today, where)


class JsonFileStore:
    """Reads the demo data once, at construction.

    Loaded eagerly and validated on the way in: a malformed fixture should stop
    the service from starting, not surface as a mysterious tool failure on the
    one ticket that happens to touch the bad record.
    """

    def __init__(self, data_dir: Path, clock: Clock) -> None:
        today = clock.now().date()
        self._orders = self._load_orders(data_dir / "orders.json", today)
        self._accounts = self._load_accounts(data_dir / "accounts.json", today)

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise ToolPermanentError(f"demo data file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ToolPermanentError(f"{path.name} is not valid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise ToolPermanentError(f"{path.name} must contain a JSON array")
        return payload

    def _load_orders(self, path: Path, today: date) -> dict[str, Order]:
        orders: dict[str, Order] = {}
        for row in self._read(path):
            where = f"{path.name}[{row.get('order_id', '?')}]"
            order = Order(
                order_id=str(row["order_id"]),
                account_id=str(row["account_id"]),
                placed_on=_offset(row, "placed_days_ago", today, where),
                total_minor=int(row["total_minor"]),
                currency=str(row["currency"]),
                status=str(row["status"]),
                items=tuple(str(item) for item in row.get("items", ())),
                condition=str(row.get("condition", "ok")),
                final_sale=bool(row.get("final_sale", False)),
                estimated_delivery_on=_optional_offset(
                    row, "estimated_delivery_days_ago", today, where
                ),
                tracking_last_moved_on=_optional_offset(
                    row, "tracking_last_moved_days_ago", today, where
                ),
            )
            if order.order_id in orders:
                raise ToolPermanentError(f"{path.name}: duplicate order_id {order.order_id}")
            orders[order.order_id] = order
        return orders

    def _load_accounts(self, path: Path, today: date) -> dict[str, Account]:
        accounts: dict[str, Account] = {}
        for row in self._read(path):
            where = f"{path.name}[{row.get('account_id', '?')}]"
            account = Account(
                account_id=str(row["account_id"]),
                plan=str(row["plan"]),
                opened_on=_offset(row, "opened_days_ago", today, where),
                locked=bool(row.get("locked", False)),
            )
            if account.account_id in accounts:
                raise ToolPermanentError(f"{path.name}: duplicate account_id {account.account_id}")
            accounts[account.account_id] = account
        return accounts

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_account(self, account_id: str) -> Account | None:
        return self._accounts.get(account_id)
