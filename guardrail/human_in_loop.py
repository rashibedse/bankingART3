"""
Backward-compatible shim.

The SENSITIVE_TOOLS set that used to be hardcoded here was the single most
banking-specific thing in the whole guardrail package — it named this
project's tools inside supposedly domain-agnostic code. It now lives in
policy.yaml under the human_in_loop policy's params.tools.

That move is also the ATK-014 fix: adding update_user_info to the guarded
set is a one-line policy edit, reviewable in a diff, rather than a code change.
"""

from __future__ import annotations

from guardrail.registry import GuardContext, sensitive_tool_call

_POLICY_ID = "human_in_loop"


def sensitive_tools(policy_path=None) -> set:
    """The currently-guarded tool set, read from the active policy."""
    from guardrail.core import load
    for p in load(policy_path).policies:
        if p.id == _POLICY_ID:
            return set(p.params.get("tools", []))
    return set()


def check(tool_calls: list, policy_path=None) -> tuple[bool, str]:
    """(triggered, reason) — triggered if any call touches a sensitive tool."""
    ctx = GuardContext(tool_calls=tool_calls or [])
    return sensitive_tool_call(ctx, {"tools": list(sensitive_tools(policy_path))})