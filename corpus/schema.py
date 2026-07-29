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
HOOKS = {"before", "after", "result", "budget"}
VECTOR_TYPES = {"prompt", "file", "scheduled_transaction"}
CHECK_TYPES = {"transfer_made", "output_contains", "call_count_over", "field_equals", "field_changed", "must_allow"}

# CI_REGRESSION_SPEC §1 metadata, used by the risk-based selector (ci_gate.py)
# and the baseline snapshot. `family` is DELIBERATELY separate from
# `category`: it's a finer taxonomy the selector's PATH_IMPACT map keys on
# (e.g. ATK-016/017 are `category: poisoned_input`/`tool_misuse` for the
# guardrail's own bookkeeping, but `family: context_poisoning`/`tool_chaining`
# for selection — a genuinely different attack shape than the rest of their
# category). "unknown" is a valid, expected value for anything not yet
# backfilled, not an error.
FAMILIES = {
    "prompt_injection", "jailbreak", "poisoned_input", "exfiltration",
    "tool_misuse", "context_poisoning", "tool_chaining", "unknown",
}
SURFACES = {"prompt", "file", "transaction", "scheduled_transaction"}
COSTS = {"cheap", "expensive"}


class AttackSchemaError(ValueError):
    """Raised when an attack YAML is missing fields or has bad values."""


class BenignSchemaError(ValueError):
    """Raised when a benign-case YAML is missing fields or has bad values."""


@dataclass
class Vector:
    type: str
    file: str | None = None
    placeholder: str | None = None
    scheduled_transaction_id: int | None = None


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
class BenignCase:
    id: str
    description: str
    user_prompt: str
    expected_tools: list[str]
    must_allow_hook: list[str]
    expect_hitl: bool
    note: str | None = None
    requires_clean_environment: bool = False


@dataclass
class MisbehaviorAction:
    tool: str
    args: dict = field(default_factory=dict)
    repeat: int = 1
    result_contains: str | None = None


@dataclass
class MisbehaviorCase:
    id: str
    category: str
    severity: str
    description: str
    user_prompt: str
    misbehavior: MisbehaviorAction
    expected_catch: dict
    success_check: SuccessCheck

    @property
    def family(self) -> str:
        """CI_REGRESSION_SPEC §1: map category -> family rather than
        duplicating it as a stored field — misbehavior cases don't need the
        finer attack taxonomy (context_poisoning/tool_chaining), so there's
        nothing for a separate field to express."""
        return self.category


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
    # No stub_action can express this attack (e.g. it depends on the model
    # reasoning over planted context, not obeying an @@ACTION@@ directive).
    # Meaningless against the deterministic stub — the runner must skip it
    # when --agent stub.
    llm_only: bool = False
    # CI_REGRESSION_SPEC §1 — selector metadata. See FAMILIES' docstring for
    # why `family` isn't just `category`.
    family: str = "unknown"
    surface: str = "prompt"
    cost: str = "cheap"  # "expensive" = needs llm_judge to be caught

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
    v = Vector(
        type=vtype,
        file=raw.get("file"),
        placeholder=raw.get("placeholder"),
        scheduled_transaction_id=raw.get("scheduled_transaction_id"),
    )
    if vtype == "file" and (not v.file or not v.placeholder):
        raise AttackSchemaError(
            f"{where}.vector: file vectors need both 'file' and 'placeholder'"
        )
    if vtype == "scheduled_transaction" and (v.scheduled_transaction_id is None or not v.placeholder):
        raise AttackSchemaError(
            f"{where}.vector: scheduled_transaction vectors need both "
            f"'scheduled_transaction_id' and 'placeholder'"
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
    if ctype == "field_changed" and not check.path:
        raise AttackSchemaError(f"{where}.success_check: field_changed needs 'path'")
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
    with path.open(encoding="utf-8") as f:
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

    family = raw.get("family", "unknown")
    if family not in FAMILIES:
        raise AttackSchemaError(f"{where}: bad family '{family}'")
    surface = raw.get("surface", "prompt")
    if surface not in SURFACES:
        raise AttackSchemaError(f"{where}: bad surface '{surface}'")
    cost = raw.get("cost", "cheap")
    if cost not in COSTS:
        raise AttackSchemaError(f"{where}: bad cost '{cost}'")

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
        llm_only=bool(raw.get("llm_only", False)),
        family=family,
        surface=surface,
        cost=cost,
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


ALLOWED_BENIGN_HOOKS = {"before", "after"}


def load_benign_case(path: str | Path) -> BenignCase:
    """Load and validate one benign-case YAML into a BenignCase object."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    where = path.name

    case_id = _require(raw, "id", where)
    tools = _require(raw, "expected_tools", where)
    if not isinstance(tools, list):
        raise BenignSchemaError(f"{where}: expected_tools must be a list")
    hooks = _require(raw, "must_allow_hook", where)
    if not isinstance(hooks, list):
        raise BenignSchemaError(f"{where}: must_allow_hook must be a list")
    for h in hooks:
        if h not in ALLOWED_BENIGN_HOOKS:
            raise BenignSchemaError(f"{where}: invalid hook '{h}' in must_allow_hook")

    return BenignCase(
        id=case_id,
        description=_require(raw, "description", where),
        user_prompt=_require(raw, "user_prompt", where),
        expected_tools=tools,
        must_allow_hook=hooks,
        expect_hitl=bool(raw.get("expect_hitl", False)),
        note=raw.get("note"),
        requires_clean_environment=bool(raw.get("requires_clean_environment", False)),
    )


def load_benign_corpus(directory: str | Path) -> list[BenignCase]:
    """Load every *.yaml in a benign directory, sorted by id."""
    directory = Path(directory)
    cases = [load_benign_case(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise BenignSchemaError(f"duplicate benign-case ids: {sorted(dupes)}")
    return cases


MISBEHAVIOR_FILE = "corpus/misbehavior.yaml"
MISBEHAVIOR_CATEGORIES = {
    "runaway_loop", "fabricated_value", "unrequested_mutation",
    "structuring_by_accident", "oversized_action", "leaked_result",
    "benign_control",
}


def _parse_misbehavior_action(raw: dict, where: str) -> MisbehaviorAction:
    tool = _require(raw, "tool", f"{where}.misbehavior")
    return MisbehaviorAction(
        tool=tool,
        args=dict(raw.get("args", {})),
        repeat=int(raw.get("repeat", 1)),
        result_contains=raw.get("result_contains"),
    )


def load_misbehavior_cases(path: str | Path = MISBEHAVIOR_FILE) -> list[MisbehaviorCase]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    where = path.name

    cases: list[MisbehaviorCase] = []
    for raw_case in raw.get("cases", []):
        cid = _require(raw_case, "id", where)
        cat = _require(raw_case, "category", f"{where}:{cid}")
        if cat not in MISBEHAVIOR_CATEGORIES:
            raise AttackSchemaError(f"{where}:{cid}: bad category '{cat}'")
        sev = raw_case.get("severity", "none")
        desc = _require(raw_case, "description", f"{where}:{cid}")
        prompt = _require(raw_case, "user_prompt", f"{where}:{cid}")
        m_action = _parse_misbehavior_action(
            _require(raw_case, "misbehavior", f"{where}:{cid}"),
            f"{where}:{cid}",
        )
        e_catch = _require(raw_case, "expected_catch", f"{where}:{cid}")
        s_check = _parse_check(
            _require(raw_case, "success_check", f"{where}:{cid}"),
            f"{where}:{cid}",
        )

        cases.append(MisbehaviorCase(
            id=cid,
            category=cat,
            severity=sev,
            description=desc,
            user_prompt=prompt,
            misbehavior=m_action,
            expected_catch=e_catch,
            success_check=s_check,
        ))

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise AttackSchemaError(f"duplicate misbehavior ids: {sorted(dupes)}")
    return cases