"""
Content filter check (before-hook component).

Keyword/pattern-based detector for prompt injection and jailbreak language.
This is the cheapest, crudest layer in the stack — real attacks can dodge
it by not using these phrases at all (see ATK-002, ATK-009, which rely on
human_in_loop as a backstop instead). It exists because it's fast, catches
the loud/obvious cases, and demonstrates that a keyword layer alone is
insufficient — which is the whole argument for stacking.

NORMALIZATION IS THE INTERESTING PART: before matching, we strip invisible
unicode characters (zero-width space/joiners, BOM). Without this step, an
attacker can break up a keyword ("in\u200bstruc\u200btions") and slip past
a naive substring match. This one normalization step is what lets this
filter catch ATK-005's evasion attempt.
"""

from __future__ import annotations

import re

# Zero-width / invisible characters attackers use to split up keywords.
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060]"
)

# (pattern, human-readable reason). Case-insensitive, matched after
# normalization. Keep patterns generic — not tied to any specific attack id.
_PATTERNS = [
    (re.compile(r"ignore (all )?(previous|prior) instructions", re.I),
     "instruction-override phrase"),
    (re.compile(r"skip (all )?verification", re.I),
     "verification-bypass phrase"),
    (re.compile(r"emergency( fraud)? mode", re.I),
     "fake-urgency framing"),
    (re.compile(r"fictional.{0,40}audit", re.I),
     "fictional-scenario jailbreak framing"),
    (re.compile(r"no real consequences", re.I),
     "consequence-free jailbreak framing"),
    (re.compile(r"system directive", re.I),
     "fake system-level directive"),
    (re.compile(r"<!--.*?-->", re.I | re.S),
     "hidden instruction in HTML/markdown comment"),
]


def normalize(text: str) -> str:
    """Strip invisible characters so obfuscated keywords resurface."""
    return _INVISIBLE_CHARS.sub("", text)


def check(text: str) -> tuple[bool, str]:
    """Return (triggered, reason). triggered=True means this text should
    be blocked; reason names which pattern fired (first match wins)."""
    clean = normalize(text)
    for pattern, reason in _PATTERNS:
        if pattern.search(clean):
            return True, reason
    return False, ""