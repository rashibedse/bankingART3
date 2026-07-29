"""
The policy engine: load YAML -> validate against the registry -> evaluate.

THIS IS THE GENERAL HALF. It knows nothing about banking, and nothing about
which checks exist — it only knows how to match a tool call against rules and
return a decision. Swapping in a different agent means a different policy.yaml,
not a different engine.

VALIDATION IS LOAD-TIME AND STRICT. A policy naming a condition, action, or
hook that doesn't exist is a hard error before anything runs — never a warning,
never a silent skip. This is what makes a generated policy.yaml safe to accept:
the generator can only produce something that passes this validator, and a
model hallucinating a condition name gets rejected at the door rather than
quietly disabling a security check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from guardrail import registry
from guardrail.registry import GuardContext


class PolicyError(ValueError):
    """Raised on any malformed or invalid policy file. Always fatal."""


@dataclass
class Policy:
    id: str
    hook: str
    condition: str
    action: str
    params: dict = field(default_factory=dict)
    layer: str = "deterministic"
    severity: str = "medium"
    description: str = ""
    enabled: bool = True


@dataclass
class PolicySet:
    name: str
    policies: list[Policy] = field(default_factory=list)
    pattern_sets: dict = field(default_factory=dict)

    def for_hook(self, hook: str) -> list[Policy]:
        """Enabled policies for one hook, in file order. File order IS
        evaluation order — a reviewer can read precedence off the page."""
        return [p for p in self.policies if p.hook == hook and p.enabled]


@dataclass
class Decision:
    """One policy's outcome. Mirrors the old CheckResult shape so existing
    JSON traces, compare.py and the dashboard keep working unchanged."""
    name: str
    triggered: bool
    reason: str = ""
    action: str = ""
    severity: str = ""
    layer: str = ""


# ---------------------------------------------------------------------
# load + validate
# ---------------------------------------------------------------------

def _validate(policy: Policy, pattern_sets: dict) -> None:
    if policy.hook not in registry.HOOKS:
        raise PolicyError(
            f"policy {policy.id!r}: unknown hook {policy.hook!r} "
            f"(valid: {sorted(registry.HOOKS)})"
        )
    if policy.condition not in registry.CONDITIONS:
        raise PolicyError(
            f"policy {policy.id!r}: unknown condition {policy.condition!r} "
            f"(valid: {sorted(registry.CONDITIONS)})"
        )
    if policy.action not in registry.ACTIONS:
        raise PolicyError(
            f"policy {policy.id!r}: unknown action {policy.action!r} "
            f"(valid: {sorted(registry.ACTIONS)})"
        )
    ref = policy.params.get("pattern_set")
    if ref is not None and ref not in pattern_sets:
        raise PolicyError(
            f"policy {policy.id!r}: references unknown pattern_set {ref!r} "
            f"(defined: {sorted(pattern_sets)})"
        )


def load_policy(path: str | Path) -> PolicySet:
    """Read and validate a policy file. Raises PolicyError on anything wrong."""
    path = Path(path)
    if not path.exists():
        raise PolicyError(f"policy file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    pattern_sets = raw.get("pattern_sets", {}) or {}
    for set_name, entries in pattern_sets.items():
        if not isinstance(entries, list):
            raise PolicyError(f"pattern_set {set_name!r} must be a list")
        for entry in entries:
            if "pattern" not in entry:
                raise PolicyError(f"pattern_set {set_name!r}: entry missing 'pattern'")

    policies: list[Policy] = []
    seen: set[str] = set()
    for item in raw.get("policies", []) or []:
        missing = {"id", "hook", "condition", "action"} - set(item)
        if missing:
            raise PolicyError(f"policy entry missing required field(s): {sorted(missing)}")
        if item["id"] in seen:
            raise PolicyError(f"duplicate policy id: {item['id']!r}")
        seen.add(item["id"])

        policy = Policy(
            id=item["id"],
            hook=item["hook"],
            condition=item["condition"],
            action=item["action"],
            params=item.get("params", {}) or {},
            layer=item.get("layer", "deterministic"),
            severity=item.get("severity", "medium"),
            description=item.get("description", ""),
            enabled=bool(item.get("enabled", True)),
        )
        _validate(policy, pattern_sets)
        policies.append(policy)

    return PolicySet(
        name=raw.get("name", path.stem),
        policies=policies,
        pattern_sets=pattern_sets,
    )


# ---------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------

_SEV_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "": 0}


def evaluate(
    policy_set: PolicySet,
    hook: str,
    ctx: GuardContext,
) -> tuple[list[Decision], Policy | None]:
    """Run every enabled policy for `hook` and return (decisions, blocker).

    ALL policies run even after one fires — the trace is meant to show which
    layers caught a given attack and which missed it, which is the whole point
    of a stacked design. `blocker` is the triggered blocking policy with the
    HIGHEST SEVERITY, not the first in file order — a CI severity gate (and a
    reviewer reading the trace) cares about the worst thing that fired, not
    which policy happened to be listed first.
    """
    ctx.pattern_sets = policy_set.pattern_sets
    decisions: list[Decision] = []
    triggered_blocking: list[Policy] = []

    for policy in policy_set.for_hook(hook):
        fn = registry.CONDITIONS[policy.condition]
        try:
            triggered, reason = fn(ctx, policy.params)
        except Exception as exc:  # noqa: BLE001
            # A broken condition must be visible, not silently non-blocking.
            triggered, reason = False, f"condition error: {type(exc).__name__}: {exc}"

        decisions.append(Decision(
            name=policy.id,
            triggered=triggered,
            reason=reason,
            action=policy.action,
            severity=policy.severity,
            layer=policy.layer,
        ))

        if triggered and policy.action in registry.BLOCKING_ACTIONS:
            triggered_blocking.append(policy)

    blocker = (
        max(triggered_blocking, key=lambda p: _SEV_RANK.get(p.severity, 0))
        if triggered_blocking else None
    )

    return decisions, blocker


def budget_for(policy_set: PolicySet) -> int | None:
    """Max tool calls allowed per invocation, from the budget-hook policy.
    None means uncapped."""
    for policy in policy_set.for_hook("budget"):
        limit = policy.params.get("max_calls")
        if limit is not None:
            return int(limit)
    return None


# ---------------------------------------------------------------------
# introspection — used to build the generator's closed vocabulary prompt
# ---------------------------------------------------------------------

def vocabulary() -> dict:
    """The closed set a policy file may reference. Pulled live from the
    registry so a policy generator is constrained by what actually exists
    in this codebase, not by what a model remembers."""
    return {
        "conditions": sorted(registry.CONDITIONS),
        "actions": sorted(registry.ACTIONS),
        "hooks": sorted(registry.HOOKS),
    }