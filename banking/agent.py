"""
The gullible stub agent — a deterministic stand-in for a real LLM.

WHY A STUB, AND WHY GULLIBLE
A real LLM reads its context and *decides* what tools to call. We can't
run one deterministically (no API here, and its choices vary run to run).
So the stub simulates the WORST-CASE model: one that complies with any
actionable instruction it finds in its context. This is deliberate — if
the guardrail can protect even a maximally-gullible agent, it protects a
cautious real one too. The stub makes the GUARDRAIL the thing under test,
not the model's judgment.

HOW A DETERMINISTIC STUB CAN BE "TRICKED"
A stub can't understand English, so an attack payload carries a tiny
machine-readable directive the stub knows how to obey, alongside the
natural-language poison. Format, one per line:

    @@ACTION tool_name arg=value; arg=value@@

The natural-language sentences around it are what a *real* model would
comply with (and what the guardrail's before-hook is meant to detect);
the @@ACTION@@ line is the stub's deterministic hook for "the model
complied." The stub obeys any @@ACTION@@ it can see.

THE GUARDRAIL SEAM
The stub is invoked by the runner ONLY if the before-guardrail allowed
the input. So:
  - before-hook blocks the poisoned input  -> stub never runs -> no
    malicious call -> attack blocked.
  - before-hook misses it -> stub sees the @@ACTION@@ -> emits the
    malicious call -> state changes -> attack succeeds.
The stub itself contains no defense. That is on purpose: defense lives in
guardrail/, never here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


ACTION_RE = re.compile(r"@@ACTION\s+(\w+)\s*(.*?)@@", re.DOTALL)


@dataclass
class ToolCall:
    """One action the agent wants to take: a tool name + its arguments."""
    name: str
    args: dict


@dataclass
class AgentResult:
    """What the agent produced this invocation."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""


def _coerce(value: str):
    """Turn a directive's string value into an int/float/bool where obvious,
    so send_money amount=1000 arrives as a number, not the text "1000"."""
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _parse_actions(text: str) -> list[ToolCall]:
    """Extract every @@ACTION ...@@ directive in text as a ToolCall."""
    calls: list[ToolCall] = []
    for name, arg_blob in ACTION_RE.findall(text):
        args: dict = {}
        for pair in arg_blob.split(";"):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            k, _, val = pair.partition("=")
            args[k.strip()] = _coerce(val)
        calls.append(ToolCall(name=name, args=args))
    return calls


def gullible_agent(user_prompt: str, context: str) -> AgentResult:
    """The stub's whole brain.

    `user_prompt` is the legitimate request. `context` is everything the
    agent has pulled in — file contents, transaction subjects, tool
    results — any of which may carry an injected @@ACTION@@ payload.

    Behavior: obey every directive found in prompt + context. A cautious
    real agent would refuse some; this gullible one refuses nothing, which
    is exactly the stress test the guardrail must survive.
    """
    blob = f"{user_prompt}\n{context}"
    calls = _parse_actions(blob)

    if calls:
        answer = "Done — carried out the requested actions."
    else:
        answer = "No actionable instructions found; nothing to do."

    return AgentResult(tool_calls=calls, final_answer=answer)