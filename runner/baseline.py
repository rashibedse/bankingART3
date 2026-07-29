"""
Baseline snapshot + regression compare — CI_REGRESSION_SPEC.md §2.

WHAT THIS IS FOR
A gate that only checks "did any attack succeed" catches a NEW BREACH but
misses a LAYER regressing quietly (still blocked, but now by a weaker/wronger
check than before). This module is what turns the corpus runner into a
*regression* suite: it snapshots what the guardrail currently does per case,
and later compares a fresh run against that snapshot to classify what
changed, not just whether it's currently green.

Two commands:
    python runner/baseline.py --snapshot [--out corpus/baseline.json]
    python runner/baseline.py --compare results/guarded.json [--baseline corpus/baseline.json]

WHY --snapshot RUNS THE ATTACKS ITSELF (not "read an existing results file")
The baseline must reflect the CURRENT corpus + CURRENT policy, not whatever
someone happened to have lying around in results/. It always runs
`--guardrail on --agent stub` — stub, because the baseline is a determinism
anchor; an LLM-mode baseline would drift on its own and manufacture false
regressions.

known_gap PRESERVATION
ATK-014 and ATK-017 are deliberately undefended right now (documented
findings, not bugs). Re-running --snapshot must not silently erase that
annotation — it's hand-authored (see CI_REGRESSION_SPEC.md §8#7) and the
whole point of a known_gap is that only a human updates it deliberately.
So --snapshot MERGES: it recomputes defended/blocked_by/blocked_at_hook/
severity from a fresh run, but carries forward known_gap/gap_reason from
whatever baseline file already sits at --out, if any.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.schema import load_corpus
from runner.run import run_attack, ENV_PATH, CORPUS_DIR  # noqa: F401  (ENV_PATH kept for parity/import-site clarity)


DEFAULT_BASELINE = "corpus/baseline.json"
POLICY_PATH = Path(__file__).resolve().parent.parent / "guardrail" / "policy.yaml"


def _policy_hash(policy_path: Path = POLICY_PATH) -> str:
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def _norm(v):
    """caught_by_hook/caught_by_check use the string "none" as their
    not-caught sentinel (see run.py); baseline.json uses JSON null. Same
    idea, different vocabulary — normalize once at the boundary."""
    return None if v in (None, "none") else v


# ---------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------

def build_snapshot(corpus_dir: str = CORPUS_DIR, existing: dict | None = None) -> dict:
    """Run every stub-eligible attack guardrail=on and build the baseline
    cases dict, preserving known_gap/gap_reason from `existing` (the
    baseline file currently on disk, if any)."""
    attacks = load_corpus(corpus_dir)
    existing_cases = (existing or {}).get("cases", {})

    cases: dict = {}
    for a in attacks:
        if a.llm_only:
            # No stub_action can express this attack — it structurally
            # cannot be run deterministically, so it has no place in a
            # stub-mode regression baseline. (RUNNER_FIXES §2 / this attack's
            # own `notes:` field say the same thing.)
            continue
        r = run_attack(a, mode="on", agent_mode="stub")
        entry = {
            "defended": not r["succeeded"],
            "blocked_by": _norm(r["caught_by_check"]),
            "blocked_at_hook": _norm(r["caught_by_hook"]),
            "severity": r["severity"],
        }
        prior = existing_cases.get(a.id, {})
        if prior.get("known_gap"):
            entry["known_gap"] = True
            entry["gap_reason"] = prior.get("gap_reason", "")
        cases[a.id] = entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": "stub",
        "guardrail": "on",
        "policy_hash": _policy_hash(),
        "cases": cases,
    }


def snapshot(out_path: str = DEFAULT_BASELINE, corpus_dir: str = CORPUS_DIR) -> dict:
    out = Path(out_path)
    existing = None
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None

    payload = build_snapshot(corpus_dir=corpus_dir, existing=existing)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------

# Regression classes — CI_REGRESSION_SPEC.md §2b, exactly the 7 rows in that
# table plus STABLE/KNOWN_GAP_OPEN for "nothing changed." `verdict` here is
# baseline.py's own opinion for standalone use; ci_gate.py applies its own
# severity-aware policy (§4) on top of these `type`s rather than trusting
# this verdict blindly (e.g. NEW_BREACH at severity=low is only a WARN to
# the gate, but it's still worth flagging loudly in a standalone compare).
#
# Deliberately NOT a class here: "undefended in baseline AND now, not marked
# known_gap." That's a standing state, not a CHANGE — this module only
# classifies drift between baseline and now. Whether every undefended,
# unmarked case got caught is the baseline-review step's job (a human reads
# the snapshot before committing it — CI_REGRESSION_SPEC.md §8#7), not
# something --compare re-litigates on every run.
_FAIL_TYPES = {"NEW_BREACH", "GAP_CLOSED"}


def _classify_case(cid: str, baseline_case: dict | None, result_row: dict | None) -> dict:
    if baseline_case is None:
        return {
            "id": cid, "type": "NEW_CASE", "verdict": "WARN",
            "severity": (result_row or {}).get("severity"),
            "detail": "present in results but not in baseline - run --snapshot to adopt it",
        }
    if result_row is None:
        return {
            "id": cid, "type": "MISSING_CASE", "verdict": "WARN",
            "severity": baseline_case.get("severity"),
            "detail": "present in baseline but not in this run - deleted, renamed, or skipped?",
        }

    severity = result_row.get("severity", baseline_case.get("severity"))
    b_defended = bool(baseline_case["defended"])
    now_defended = not result_row["succeeded"]
    now_blocked_by = _norm(result_row.get("caught_by_check"))
    now_blocked_at = _norm(result_row.get("caught_by_hook"))
    known_gap = bool(baseline_case.get("known_gap", False))

    if known_gap:
        if now_defended:
            return {
                "id": cid, "type": "GAP_CLOSED", "verdict": "FAIL", "severity": severity,
                "detail": (
                    f"documented gap ({baseline_case.get('gap_reason', 'no reason on file')}) "
                    f"is now defended at {now_blocked_at}/{now_blocked_by} - update baseline.json "
                    f"deliberately (this is good news, but it must be an intentional edit, not a "
                    f"silent side effect of an unrelated change)"
                ),
            }
        return {
            "id": cid, "type": "KNOWN_GAP_OPEN", "verdict": "INFO", "severity": severity,
            "detail": baseline_case.get("gap_reason", ""),
        }

    if b_defended and not now_defended:
        return {
            "id": cid, "type": "NEW_BREACH", "verdict": "FAIL", "severity": severity,
            "detail": f"was defended ({baseline_case.get('blocked_at_hook')}/{baseline_case.get('blocked_by')}), now succeeds",
        }
    if not b_defended and now_defended:
        return {
            "id": cid, "type": "FIXED", "verdict": "PASS", "severity": severity,
            "detail": f"was undefended, now caught at {now_blocked_at}/{now_blocked_by} - consider --snapshot to adopt",
        }
    if not b_defended and not now_defended:
        # Unchanged: undefended before, undefended now, not a documented
        # known_gap. Not this module's job to police (see note above
        # _FAIL_TYPES) — surfaced as STABLE so a plain compare stays quiet
        # on it, same as any other no-drift case.
        return {"id": cid, "type": "STABLE", "verdict": "OK", "severity": severity,
                "detail": "undefended, unchanged, not marked known_gap"}

    # both defended
    if baseline_case.get("blocked_by") != now_blocked_by or baseline_case.get("blocked_at_hook") != now_blocked_at:
        return {
            "id": cid, "type": "LAYER_DRIFT", "verdict": "WARN", "severity": severity,
            "detail": f"was {baseline_case.get('blocked_at_hook')}/{baseline_case.get('blocked_by')}, now {now_blocked_at}/{now_blocked_by}",
        }
    return {"id": cid, "type": "STABLE", "verdict": "OK", "severity": severity, "detail": ""}


def compare(results_path: str, baseline_path: str = DEFAULT_BASELINE) -> dict:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    bpath = Path(baseline_path)
    if bpath.exists():
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
    else:
        # No baseline yet (e.g. a fresh clone before the first `--snapshot`
        # + commit — CI_REGRESSION_SPEC.md §8#7 is a human task, so this is
        # an expected state, not an error). Every case falls out as
        # NEW_CASE (WARN) rather than crashing the gate.
        baseline = {"generated_at": None, "policy_hash": None, "cases": {}}

    result_idx = {r["id"]: r for r in results["results"]}
    baseline_cases = baseline.get("cases", {})

    ids = sorted(set(result_idx) | set(baseline_cases))
    cases = [
        _classify_case(cid, baseline_cases.get(cid), result_idx.get(cid))
        for cid in ids
    ]

    summary: dict[str, int] = {}
    for c in cases:
        summary[c["type"]] = summary.get(c["type"], 0) + 1

    current_hash = _policy_hash()
    baseline_hash = baseline.get("policy_hash")
    policy_drift = current_hash != baseline_hash

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_file": results_path,
        "baseline_file": baseline_path,
        "baseline_generated_at": baseline.get("generated_at"),
        "policy_hash": current_hash,
        "baseline_policy_hash": baseline_hash,
        "policy_drift": policy_drift,
        "summary": summary,
        "cases": cases,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _print_compare(report: dict) -> None:
    noisy = [c for c in report["cases"] if c["type"] not in ("STABLE",)]
    for c in sorted(noisy, key=lambda c: c["id"]):
        print(f"  {c['id']:9} {c['type']:16} {c['verdict']:5} {c['detail']}")
    if report["policy_drift"]:
        print(f"  POLICY_DRIFT: policy_hash changed "
              f"({report['baseline_policy_hash']} -> {report['policy_hash']})")
    s = report["summary"]
    print(f"\n{s.get('STABLE', 0)} stable, {s.get('KNOWN_GAP_OPEN', 0)} known gap(s) open, "
          + ", ".join(f"{v} {k}" for k, v in sorted(s.items())
                       if k not in ("STABLE", "KNOWN_GAP_OPEN")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline snapshot + regression compare.")
    parser.add_argument("--snapshot", action="store_true", help="write current stub/guarded results as the baseline")
    parser.add_argument("--compare", metavar="RESULTS_JSON", default=None,
                         help="diff a results JSON (e.g. results/guarded.json) against the baseline")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline file path")
    parser.add_argument("--corpus", default=CORPUS_DIR, help="attack corpus dir (snapshot only)")
    args = parser.parse_args()

    if not args.snapshot and not args.compare:
        parser.error("pass --snapshot or --compare RESULTS_JSON")

    if args.snapshot:
        payload = snapshot(out_path=args.baseline, corpus_dir=args.corpus)
        n = len(payload["cases"])
        gaps = sum(1 for c in payload["cases"].values() if c.get("known_gap"))
        print(f"Snapshotted {n} cases ({gaps} known gap(s) preserved) -> {args.baseline}")
        return

    report = compare(args.compare, baseline_path=args.baseline)
    _print_compare(report)
    fail = any(c["type"] in _FAIL_TYPES for c in report["cases"])
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
