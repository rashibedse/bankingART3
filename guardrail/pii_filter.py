"""
PII check — used on both sides of the stack with different behavior:

  before_guard (input)  -> mode="redact": mask PII, let the (redacted)
                            text continue into the agent. Soft touch —
                            we don't yet know if this PII matters.
  after_guard  (output)  -> mode="block": if PII/secrets are about to
                            leave the agent, refuse outright. By this
                            point a leak is a leak — no soft option.

This is a generic detector: IBAN-shaped account numbers and a "password"
heuristic. It knows nothing about banking specifically — an IBAN pattern
is a general-purpose PII shape, not a banking concept, and the same check
would fire on any domain that happens to contain one.
"""

from __future__ import annotations

import re

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# heuristic: the word "password" appearing at all in agent-generated output
# is treated as a secret-leak signal — a banking agent's output should never
# need to say it, redacted value or not.
_PASSWORD_MENTION = re.compile(r"\bpassword\b", re.I)


def _find(text: str) -> tuple[bool, str]:
    """Return (found, reason) — does NOT redact, just detects."""
    if _IBAN.search(text):
        return True, "IBAN-shaped account number detected"
    if _SSN.search(text):
        return True, "SSN-shaped value detected"
    if _PASSWORD_MENTION.search(text):
        return True, "password reference detected"
    return False, ""


def redact(text: str) -> tuple[str, bool, str]:
    """Input-side: mask PII, return (redacted_text, triggered, reason).
    Never blocks — the request continues with the masked text."""
    triggered, reason = _find(text)
    if not triggered:
        return text, False, ""
    redacted = _IBAN.sub("[REDACTED-IBAN]", text)
    redacted = _SSN.sub("[REDACTED-SSN]", redacted)
    return redacted, True, reason


def check(text: str) -> tuple[bool, str]:
    """Output-side: pure detect, for the caller to block on. No mutation —
    once it's in the output, masking it after the fact isn't good enough."""
    return _find(text)