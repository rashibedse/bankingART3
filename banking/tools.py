"""
Banking tools: the actions the agent can take on the environment.

Each tool is a plain function whose FIRST argument is the Environment it
acts on; the rest are the arguments the agent supplies. Read tools return
a value the agent "sees"; write tools mutate the environment and return a
short confirmation string.

Every tool is registered in TOOLS (name -> function). This registry is
what makes execution generic: the runner receives an agent's chosen call
as (name, args) and does TOOLS[name](env, **args) without hard-coding any
specific tool. That generic shape is also exactly what a guardrail sees —
"a call named X with args Y" — never a banking-specific branch.

read_file is the single most attack-relevant tool: it returns raw file
contents straight into the agent's context, which is how a poisoned file
(bill, landlord notice) delivers its payload. The tool itself does no
filtering — catching poison is the guardrail's job, not the tool's.
"""

from __future__ import annotations

from .environment import Environment, Transaction


# ---- read tools -------------------------------------------------------

def get_balance(env: Environment) -> float:
    """Return the current account balance."""
    return env.bank_account.balance


def get_iban(env: Environment) -> str:
    """Return the user's own IBAN (account number)."""
    return env.bank_account.iban


def get_most_recent_transactions(env: Environment, n: int = 100) -> list[dict]:
    """Return the n most recent past transactions, newest last."""
    txns = env.bank_account.transactions[-n:]
    return [_txn_to_dict(t) for t in txns]


def get_scheduled_transactions(env: Environment) -> list[dict]:
    """Return all scheduled (future) transactions."""
    return [_txn_to_dict(t) for t in env.bank_account.scheduled_transactions]


def read_file(env: Environment, filename: str) -> str:
    """Return the raw contents of a file, or an error string if missing.

    Returns file contents verbatim — including any injected payload. This
    is the primary poisoned-input delivery path into the agent's context.
    """
    if filename not in env.files:
        return f"[error] no such file: {filename}"
    return env.files[filename]


def get_user_info(env: Environment) -> dict:
    """Return the user's profile (no password)."""
    u = env.user_account
    return {
        "first_name": u.first_name,
        "last_name": u.last_name,
        "street": u.street,
        "city": u.city,
    }


# ---- write tools ------------------------------------------------------

def send_money(
    env: Environment,
    recipient: str,
    amount: float,
    subject: str,
    date: str,
) -> str:
    """Send money now: append a transaction and debit the balance."""
    amount = float(amount)
    txn = Transaction(
        id=env.next_transaction_id(),
        sender="me",
        recipient=recipient,
        amount=amount,
        subject=subject,
        date=date,
        recurring=False,
    )
    env.bank_account.transactions.append(txn)
    env.bank_account.balance -= amount
    return f"[ok] sent {amount} to {recipient} (subject: {subject})"


def schedule_transaction(
    env: Environment,
    recipient: str,
    amount: float,
    subject: str,
    date: str,
    recurring: bool = False,
) -> str:
    """Schedule a future transaction (no immediate balance change)."""
    amount = float(amount)
    txn = Transaction(
        id=env.next_transaction_id(),
        sender=env.bank_account.iban,
        recipient=recipient,
        amount=amount,
        subject=subject,
        date=date,
        recurring=bool(recurring),
    )
    env.bank_account.scheduled_transactions.append(txn)
    return f"[ok] scheduled {amount} to {recipient} on {date} (recurring: {recurring})"


def update_scheduled_transaction(
    env: Environment,
    id: int,
    recipient: str | None = None,
    amount: float | None = None,
    subject: str | None = None,
    date: str | None = None,
    recurring: bool | None = None,
) -> str:
    """Modify an existing scheduled transaction by id."""
    for t in env.bank_account.scheduled_transactions:
        if t.id == id:
            if recipient is not None:
                t.recipient = recipient
            if amount is not None:
                t.amount = float(amount)
            if subject is not None:
                t.subject = subject
            if date is not None:
                t.date = date
            if recurring is not None:
                t.recurring = bool(recurring)
            return f"[ok] updated scheduled transaction {id}"
    return f"[error] no scheduled transaction with id {id}"


def update_user_info(
    env: Environment,
    first_name: str | None = None,
    last_name: str | None = None,
    street: str | None = None,
    city: str | None = None,
) -> str:
    """Update the user's profile fields."""
    u = env.user_account
    if first_name is not None:
        u.first_name = first_name
    if last_name is not None:
        u.last_name = last_name
    if street is not None:
        u.street = street
    if city is not None:
        u.city = city
    return "[ok] updated user info"


def update_password(env: Environment, password: str) -> str:
    """Set a new account password."""
    env.user_account.password = password
    return "[ok] password updated"


# ---- helpers + registry ----------------------------------------------

def _txn_to_dict(t: Transaction) -> dict:
    return {
        "id": t.id,
        "sender": t.sender,
        "recipient": t.recipient,
        "amount": t.amount,
        "subject": t.subject,
        "date": t.date,
        "recurring": t.recurring,
    }


# name -> function. The runner executes an agent's chosen call via this
# table; adding a tool here never requires touching runner or guardrail.
TOOLS = {
    "get_balance": get_balance,
    "get_iban": get_iban,
    "get_most_recent_transactions": get_most_recent_transactions,
    "get_scheduled_transactions": get_scheduled_transactions,
    "read_file": read_file,
    "get_user_info": get_user_info,
    "send_money": send_money,
    "schedule_transaction": schedule_transaction,
    "update_scheduled_transaction": update_scheduled_transaction,
    "update_user_info": update_user_info,
    "update_password": update_password,
}