"""
The guardrail seam — now policy-driven.

WHAT CHANGED: before_guard/after_guard no longer hardcode which checks run
in which order. They load a policy file and iterate whatever it declares for
that hook. Adding a layer, reordering precedence, or closing a coverage gap
(e.g. putting update_user_info under human-in-the-loop) is now a config diff
a reviewer can read — not a code change.

WHAT DELIBERATELY DID NOT CHANGE: the public signatures
    before_guard(text, mode)          -> Verdict
    after_guard(text, tool_calls, mode) -> Verdict
    budget_limit(mode)                -> int | None
and the check names appearing in the trace (content_filter, pii_redact,
pii_block, human_in_loop). The runner, compare.py and the dashboard keep
working untouched. All the change is internal.

Still imports NOTHING from banking/ or corpus/. Every check operates on plain
text and a generic tool-call shape (name + args), which is what keeps this
package framework- and domain-agnostic — and what makes "point it at a
different agent" a policy-file question rather than a rewrite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from guardrail import policy as policy_mod
from guardrail.registry import GuardContext

# Default policy location. Override per-run with the `policy_path` argument
# or the GUARDRAIL_POLICY env var — this is the seam the CI runner uses to
# test a candidate policy against the corpus before it merges.
DEFAULT_POLICY = Path(__file__).resolve().parent / "policy.yaml"


@dataclass
class CheckResult:
    """Unchanged shape — existing JSON traces depend on it."""
    name: str
    triggered: bool
    reason: str = ""
    action: str = ""
    severity: str = ""
    layer: str = ""


@dataclass
class Verdict:
    allow: bool
    hook: str = ""
    checks: list = field(default_factory=list)
    text: str = ""          # possibly-redacted text (before_guard only)
    blocked_by: str = ""    # which policy id caused the block, if any
    severity: str = ""      # severity of the blocking policy — drives CI gating


_CACHE: dict = {}


def load(policy_path: str | Path | None = None):
    """Load and cache a policy set. Cached by resolved path, so a run that
    swaps policies mid-flight gets the right one."""
    path = Path(policy_path or os.environ.get("GUARDRAIL_POLICY") or DEFAULT_POLICY)
    key = str(path.resolve())
    if key not in _CACHE:
        _CACHE[key] = policy_mod.load_policy(path)
    return _CACHE[key]


def _to_check(decision) -> CheckResult:
    return CheckResult(
        name=decision.name,
        triggered=decision.triggered,
        reason=decision.reason,
        action=decision.action,
        severity=decision.severity,
        layer=decision.layer,
    )


def before_guard(text: str, mode: str = "off", policy_path=None) -> Verdict:
    """Guard the INPUT entering the agent."""
    if mode == "off":
        return Verdict(allow=True, hook="before", text=text)

    ps = load(policy_path)
    ctx = GuardContext(text=text)
    decisions, blocker = policy_mod.evaluate(ps, "before", ctx)
    checks = [_to_check(d) for d in decisions]

    # Transforming conditions (pii_redact) hand back modified text.
    out_text = ctx.redacted_text if ctx.redacted_text is not None else text

    if blocker is not None:
        # On a block the caller sees the ORIGINAL text — redaction is for
        # text that continues into the agent, and nothing is continuing.
        return Verdict(allow=False, hook="before", checks=checks,
                       text=text, blocked_by=blocker.id, severity=blocker.severity)

    return Verdict(allow=True, hook="before", checks=checks, text=out_text)


def after_guard(text: str, tool_calls: list, mode: str = "off", policy_path=None) -> Verdict:
    """Guard the OUTPUT leaving the agent, plus its decided tool calls."""
    if mode == "off":
        return Verdict(allow=True, hook="after", text=text)

    ps = load(policy_path)
    ctx = GuardContext(text=text, tool_calls=tool_calls or [])
    decisions, blocker = policy_mod.evaluate(ps, "after", ctx)
    checks = [_to_check(d) for d in decisions]

    if blocker is not None:
        return Verdict(allow=False, hook="after", checks=checks,
                       blocked_by=blocker.id, severity=blocker.severity)

    return Verdict(allow=True, hook="after", checks=checks, text=text)


def on_result(tool_name: str, result, mode: str = "off", policy_path=None) -> Verdict:
    """Guard a single tool's RESULT before it re-enters agent context.

    This is the real exfiltration seam: it sees what a tool returned, not
    just that it was called."""
    if mode == "off":
        return Verdict(allow=True, hook="result", text=str(result))

    ps = load(policy_path)
    ctx = GuardContext(text=str(result), tool_name=tool_name, tool_result=str(result))
    decisions, blocker = policy_mod.evaluate(ps, "result", ctx)
    checks = [_to_check(d) for d in decisions]

    if blocker is not None:
        return Verdict(allow=False, hook="result", checks=checks,
                       blocked_by=blocker.id, severity=blocker.severity, text=str(result))

    return Verdict(allow=True, hook="result", checks=checks, text=str(result))


def budget_limit(mode: str = "off", policy_path=None) -> int | None:
    """Max tool calls the agent may make in one invocation. None = no cap."""
    if mode == "off":
        return None
    return policy_mod.budget_for(load(policy_path))


def describe(policy_path=None) -> dict:
    """What this guardrail is currently enforcing. Useful for the dashboard
    and for the pitch: the enforced policy is inspectable, not implicit."""
    ps = load(policy_path)
    return {
        "name": ps.name,
        "policies": [
            {"id": p.id, "hook": p.hook, "condition": p.condition,
             "action": p.action, "severity": p.severity,
             "layer": p.layer, "enabled": p.enabled}
            for p in ps.policies
        ],
    }