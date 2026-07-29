"""
Backward-compatible shim. Patterns now live in policy.yaml under
pattern_sets.pii_shapes; logic lives in registry.pii_detect / pii_redact.
See content_filter.py for why the split.
"""

from __future__ import annotations

from guardrail.registry import GuardContext, pii_detect, pii_redact

_SET = "pii_shapes"


def _ctx(text: str, policy_path=None):
    from guardrail.core import load
    ps = load(policy_path)
    return GuardContext(text=text, pattern_sets=ps.pattern_sets), ps


def check(text: str, policy_path=None) -> tuple[bool, str]:
    """Output-side: pure detect, for the caller to block on."""
    ctx, ps = _ctx(text, policy_path)
    if _SET not in ps.pattern_sets:
        return False, ""
    return pii_detect(ctx, {"pattern_set": _SET})


def redact(text: str, policy_path=None) -> tuple[str, bool, str]:
    """Input-side: mask PII, return (redacted_text, triggered, reason)."""
    ctx, ps = _ctx(text, policy_path)
    if _SET not in ps.pattern_sets:
        return text, False, ""
    triggered, reason = pii_redact(ctx, {"pattern_set": _SET})
    return (ctx.redacted_text if ctx.redacted_text is not None else text), triggered, reason