"""
Attack corpus schema + loader.

WHAT THIS IS
Each attack is a YAML file. This module defines the SHAPE those files must
have, validates them, and loads them into typed Attack objects. It does
NOT run attacks (that's the runner) and knows nothing about the guardrail.
Think of it as the blank form template + the form checker.

THE SHAPE OF ONE ATTACK
  id           unique label, e.g. ATK-001 (stable — the baseline tracks it)
  category     prompt_injection | jailbreak | poisoned_input |
               exfiltration | tool_misuse
  severity     low | medium | high | critical  (drives the CI gate later)
  description  one human sentence
  expected_hook  which guardrail layer SHOULD catch it:
                 before (input guard) | after (output guard) | budget
  vector       WHERE the poison enters:
                 type: prompt  -> poison rides in the user's message
                 type: file    -> poison is planted in a file the agent
                                  reads (indirect injection); needs `file`
                                  and `placeholder`
  user_prompt  the legitimate-looking message the user sends
  payload      the NATURAL-LANGUAGE malicious text. This is the realistic
               attack and the ONLY thing the before-guard is allowed to
               detect. (A real attacker writes exactly this.)
  stub_action  OPTIONAL. The action a fully-compliant agent would take,
               which the deterministic stub carries out via @@ACTION@@:
                 tool:   tool name
                 args:   {arg: value}
                 repeat: how many times (default 1; >1 = flood, for budget)
               Omit for pure-framing attacks that request no concrete tool
               action. In real-LLM mode this field is simply ignored.
  success_check  HOW the runner decides the attack SUCCEEDED (defense
                 failed). One of:
                 - type: transfer_made   recipient: ...  [amount: ...]
                     an outbound transaction to `recipient` exists
                 - type: output_contains value: ...
                     the agent's output carries a forbidden string (leak)
                 - type: call_count_over limit: N
                     the agent emitted more than N tool calls (flood)

WHY payload AND stub_action ARE SEPARATE
`payload` is the single source of truth for the *attack* — it's what the
guardrail must catch and it's identical in stub and LLM mode. `stub_action`
is scaffolding: it exists only so the dumb stub can *act out* compliance.
Drop the stub for a real model and stub_action goes unused; the payload
(the real attack) is unchanged. Keeping them apart is what lets the same
corpus serve both modes honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "poisoned_input",
    "exfiltration",
    "tool_misuse",
}
SEVERITIES = {"low", "medium", "high", "critical"}
HOOKS = {"before", "after", "budget"}
VECTOR_TYPES = {"prompt", "file"}
CHECK_TYPES = {"transfer_made", "output_contains", "call_count_over", "field_equals"}


class AttackSchemaError(ValueError):
    """Raised when an attack YAML is missing fields or has bad values."""


@dataclass
class Vector:
    type: str
    file: str | None = None
    placeholder: str | None = None


@dataclass
class StubAction:
    tool: str
    args: dict = field(default_factory=dict)
    repeat: int = 1


@dataclass
class SuccessCheck:
    type: str
    # transfer_made
    recipient: str | None = None
    amount: float | None = None
    # output_contains
    value: str | None = None
    # call_count_over
    limit: int | None = None
    # field_equals: dotted path into the Environment, e.g. "user_account.street"
    path: str | None = None


@dataclass
class Attack:
    id: str
    category: str
    severity: str
    description: str
    expected_hook: str
    vector: Vector
    user_prompt: str
    payload: str
    success_check: SuccessCheck
    stub_action: StubAction | None = None
    pr_subset: bool = False  # part of the fast CI subset?

    def build_action_text(self) -> str:
        """Turn stub_action into the @@ACTION@@ line(s) the stub obeys.

        Returns "" when there's no stub_action (pure-framing attack) or when
        you'd run in LLM mode. This is the ONLY place the @@ACTION@@ format
        is produced — the guardrail never sees or produces it.
        """
        if self.stub_action is None:
            return ""
        arg_str = "; ".join(f"{k}={v}" for k, v in self.stub_action.args.items())
        line = f"@@ACTION {self.stub_action.tool} {arg_str}@@"
        return "\n".join([line] * max(1, self.stub_action.repeat))

    def poison_text(self) -> str:
        """The full injected blob: natural-language payload + (stub mode) the
        @@ACTION@@ scaffolding. This is what gets placed into the vector."""
        action = self.build_action_text()
        return f"{self.payload}\n{action}".strip() if action else self.payload


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise AttackSchemaError(f"{where}: missing required field '{key}'")
    return d[key]


def _parse_vector(raw: dict, where: str) -> Vector:
    vtype = _require(raw, "type", f"{where}.vector")
    if vtype not in VECTOR_TYPES:
        raise AttackSchemaError(f"{where}.vector: bad type '{vtype}'")
    v = Vector(type=vtype, file=raw.get("file"), placeholder=raw.get("placeholder"))
    if vtype == "file" and (not v.file or not v.placeholder):
        raise AttackSchemaError(
            f"{where}.vector: file vectors need both 'file' and 'placeholder'"
        )
    return v


def _parse_check(raw: dict, where: str) -> SuccessCheck:
    ctype = _require(raw, "type", f"{where}.success_check")
    if ctype not in CHECK_TYPES:
        raise AttackSchemaError(f"{where}.success_check: bad type '{ctype}'")
    check = SuccessCheck(
        type=ctype,
        recipient=raw.get("recipient"),
        amount=(float(raw["amount"]) if "amount" in raw else None),
        value=raw.get("value"),
        limit=(int(raw["limit"]) if "limit" in raw else None),
        path=raw.get("path"),
    )
    if ctype == "transfer_made" and not check.recipient:
        raise AttackSchemaError(f"{where}.success_check: transfer_made needs 'recipient'")
    if ctype == "output_contains" and check.value is None:
        raise AttackSchemaError(f"{where}.success_check: output_contains needs 'value'")
    if ctype == "call_count_over" and check.limit is None:
        raise AttackSchemaError(f"{where}.success_check: call_count_over needs 'limit'")
    if ctype == "field_equals" and (not check.path or check.value is None):
        raise AttackSchemaError(f"{where}.success_check: field_equals needs 'path' and 'value'")
    return check


def _parse_action(raw: dict | None, where: str) -> StubAction | None:
    if raw is None:
        return None
    tool = _require(raw, "tool", f"{where}.stub_action")
    return StubAction(
        tool=tool,
        args=dict(raw.get("args", {})),
        repeat=int(raw.get("repeat", 1)),
    )


def load_attack(path: str | Path) -> Attack:
    """Load and validate one attack YAML into an Attack object."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    where = path.name

    category = _require(raw, "category", where)
    if category not in CATEGORIES:
        raise AttackSchemaError(f"{where}: bad category '{category}'")
    severity = _require(raw, "severity", where)
    if severity not in SEVERITIES:
        raise AttackSchemaError(f"{where}: bad severity '{severity}'")
    hook = _require(raw, "expected_hook", where)
    if hook not in HOOKS:
        raise AttackSchemaError(f"{where}: bad expected_hook '{hook}'")

    return Attack(
        id=_require(raw, "id", where),
        category=category,
        severity=severity,
        description=_require(raw, "description", where),
        expected_hook=hook,
        vector=_parse_vector(_require(raw, "vector", where), where),
        user_prompt=_require(raw, "user_prompt", where),
        payload=_require(raw, "payload", where),
        success_check=_parse_check(_require(raw, "success_check", where), where),
        stub_action=_parse_action(raw.get("stub_action"), where),
        pr_subset=bool(raw.get("pr_subset", False)),
    )


def load_corpus(directory: str | Path) -> list[Attack]:
    """Load every *.yaml in a directory, sorted by id for stable ordering."""
    directory = Path(directory)
    attacks = [load_attack(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [a.id for a in attacks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise AttackSchemaError(f"duplicate attack ids: {sorted(dupes)}")
    return attacks