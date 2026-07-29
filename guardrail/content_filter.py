"""
DEPRECATED as a policy owner — kept as a backward-compatible shim.

The patterns that used to live here now live in policy.yaml under
pattern_sets.injection_phrases, and the matching logic lives in
registry.content_scan. This module remains only so any existing import
(`from guardrail import content_filter; content_filter.check(text)`)
keeps working while the rest of the codebase migrates.

WHY THE MOVE: this module WAS the policy — the pattern list and the engine
were the same file, so tightening detection meant editing Python. Splitting
data (policy.yaml) from mechanism (registry.py) is what makes the toolkit
reusable across agents and makes a policy change a reviewable config diff.
"""

from __future__ import annotations

from guardrail.registry import GuardContext, normalize, content_scan  # noqa: F401

_SET = "injection_phrases"


def check(text: str, policy_path=None) -> tuple[bool, str]:
    """(triggered, reason) — now sourced from the active policy file."""
    from guardrail.core import load
    ps = load(policy_path)
    ctx = GuardContext(text=text, pattern_sets=ps.pattern_sets)
    if _SET not in ps.pattern_sets:
        return False, ""
    return content_scan(ctx, {"pattern_set": _SET})