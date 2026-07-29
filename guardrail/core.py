"""
The guardrail seam - now with real stacked checks.

STACKING: each hook runs a LIST of independent checks. Any check that
blocks makes the whole hook block. Each check's result is recorded, so the
JSON trace shows exactly which layer caught (or missed) an attack - not
just "before" or "after", but "content_filter" vs "pii" vs "human_in_loop".

  before_guard(text):
    1. content_filter - keyword/pattern match (with invisible-char
       normalization, so it survives simple obfuscation)
    2. pii_filter (redact mode) - masks PII, does not block; the
       (possibly redacted) text is returned for the agent to use

  after_guard(text, tool_calls):
    1. pii_filter (block mode) - a secret leaving is a hard stop
    2. human_in_loop - any sensitive tool call requires approval,
       which (per policy, no human present) is scored as a block

  budget_limit(mode) - unchanged: caps tool call count, catches floods.

Still imports NOTHING from banking/ or corpus/ - every check operates on
plain text and a generic tool-call shape (name + args), which is what
keeps this whole package framework- and domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guardrail import content_filter, pii_filter, human_in_loop


@dataclass
class CheckResult:
    name: str
    triggered: bool
    reason: str = ""


@dataclass
class Verdict:
    allow: bool
    hook: str = ""
    checks: list = field(default_factory=list)
    text: str = ""          # possibly-redacted text (before_guard only)
    blocked_by: str = ""    # which check name caused the block, if any


def before_guard(text: str, mode: str = "off") -> Verdict:
    """Guard the INPUT entering the agent: content filter + PII redaction."""
    if mode == "off":
        return Verdict(allow=True, hook="before", text=text)

    checks = []

    cf_triggered, cf_reason = content_filter.check(text)
    checks.append(CheckResult("content_filter", cf_triggered, cf_reason))

    redacted_text, pii_triggered, pii_reason = pii_filter.redact(text)
    checks.append(CheckResult("pii_redact", pii_triggered, pii_reason))

    if cf_triggered:
        return Verdict(allow=False, hook="before", checks=checks,
                        text=text, blocked_by="content_filter")

    # PII redaction never blocks on input - it masks and continues.
    return Verdict(allow=True, hook="before", checks=checks, text=redacted_text)


def after_guard(text: str, tool_calls: list, mode: str = "off") -> Verdict:
    """Guard the OUTPUT leaving the agent: PII block + sensitive-tool approval."""
    if mode == "off":
        return Verdict(allow=True, hook="after", text=text)

    checks = []

    pii_triggered, pii_reason = pii_filter.check(text)
    checks.append(CheckResult("pii_block", pii_triggered, pii_reason))

    hitl_triggered, hitl_reason = human_in_loop.check(tool_calls)
    checks.append(CheckResult("human_in_loop", hitl_triggered, hitl_reason))

    if pii_triggered:
        return Verdict(allow=False, hook="after", checks=checks, blocked_by="pii_block")
    if hitl_triggered:
        return Verdict(allow=False, hook="after", checks=checks, blocked_by="human_in_loop")

    return Verdict(allow=True, hook="after", checks=checks, text=text)


def budget_limit(mode: str = "off") -> int | None:
    """Max tool calls the agent may make in one invocation. None = no cap."""
    if mode == "off":
        return None
    return 5