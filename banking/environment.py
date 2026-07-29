"""
The banking environment: the fake bank's *state*.

This module loads environment.yaml into plain, mutable Python objects.
Nothing here knows about attacks, guardrails, or agents — it is purely
the world the agent operates on. Tools (tools.py) read and mutate this
state; success_checks (in the runner, later) inspect it after a run to
decide whether an attack changed the world it shouldn't have.

Design notes:
- Dataclasses, not dicts, so field names are explicit and typo-proof.
- load_environment() returns a FRESH Environment every call. Each attack
  must run against a clean bank — otherwise attack N sees the damage
  from attack N-1 and results become order-dependent. Freshness is the
  whole reason the loader exists instead of a module-level global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Transaction:
    id: int
    sender: str
    recipient: str
    amount: float
    subject: str
    date: str
    recurring: bool


@dataclass
class UserAccount:
    first_name: str
    last_name: str
    street: str
    city: str
    password: str


@dataclass
class BankAccount:
    balance: float
    iban: str
    transactions: list[Transaction] = field(default_factory=list)
    scheduled_transactions: list[Transaction] = field(default_factory=list)


@dataclass
class Environment:
    bank_account: BankAccount
    user_account: UserAccount
    files: dict[str, str] = field(default_factory=dict)

    def next_transaction_id(self) -> int:
        """Next free id across both transaction lists, so new sends/schedules
        never collide with an existing id."""
        ids = [t.id for t in self.bank_account.transactions]
        ids += [t.id for t in self.bank_account.scheduled_transactions]
        return (max(ids) + 1) if ids else 1


def _load_transactions(raw: list[dict]) -> list[Transaction]:
    return [
        Transaction(
            id=t["id"],
            sender=t["sender"],
            recipient=t["recipient"],
            amount=float(t["amount"]),
            subject=t["subject"],
            date=t["date"],
            recurring=bool(t["recurring"]),
        )
        for t in raw
    ]


def load_environment(path: str | Path) -> Environment:
    """Read environment.yaml into a fresh Environment.

    NOTE: this loads the environment with injection placeholders still
    UNRESOLVED (e.g. a transaction subject may literally contain the text
    "{injection_incoming_transaction}"). Substituting an attack payload
    into those placeholders is the runner's job, done per-attack right
    after loading — not here. Keeping load and injection separate means
    this loader stays attack-agnostic.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ba = raw["bank_account"]
    bank_account = BankAccount(
        balance=float(ba["balance"]),
        iban=ba["iban"],
        transactions=_load_transactions(ba.get("transactions", [])),
        scheduled_transactions=_load_transactions(ba.get("scheduled_transactions", [])),
    )

    ua = raw["user_account"]
    user_account = UserAccount(
        first_name=ua["first_name"],
        last_name=ua["last_name"],
        street=ua["street"],
        city=ua["city"],
        password=ua["password"],
    )

    files = dict(raw.get("filesystem", {}).get("files", {}))

    return Environment(
        bank_account=bank_account,
        user_account=user_account,
        files=files,
    )