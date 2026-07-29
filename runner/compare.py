"""
compare.py — the before/after diff between an unguarded and a guarded run.

Reads results/baseline.json (--guardrail off) and results/guarded.json
(--guardrail on), and classifies each attack:

  GAP          baseline succeeded, guarded ALSO succeeded
               -> the guardrail did not stop this attack. The most
               important row in the whole report.
  DEFENDED     baseline succeeded, guarded was defended
               -> the guardrail is doing its job. Records which hook
               and which specific check caught it.
  ANOMALY      baseline was defended (shouldn't happen — baseline has
               no guardrail active) — flagged in case something is off
               with a specific attack's setup, not a guardrail result.

This mirrors the classic CI security-gate framing (Snyk/Dependabot-style):
GAP rows are what should fail a build; DEFENDED rows are the proof the
shield works. Pure data producer — writes results/compare.json, one line per
attack as a heartbeat. No tables, no formatting decisions; Streamlit and
the CI gate read the JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _index(payload: dict) -> dict:
    return {r["id"]: r for r in payload["results"]}


def compare(baseline_path: str, guarded_path: str) -> dict:
    baseline = _load(baseline_path)
    guarded = _load(guarded_path)
    b_idx = _index(baseline)
    g_idx = _index(guarded)

    ids = sorted(set(b_idx) | set(g_idx))
    rows = []

    for aid in ids:
        b = b_idx.get(aid)
        g = g_idx.get(aid)

        if b is None or g is None:
            rows.append({
                "id": aid,
                "verdict": "MISSING",
                "note": "present in only one run — corpus changed between runs",
            })
            continue

        b_succeeded = b["succeeded"]
        g_succeeded = g["succeeded"]

        if b_succeeded and g_succeeded:
            verdict = "GAP"
        elif b_succeeded and not g_succeeded:
            verdict = "DEFENDED"
        elif not b_succeeded and g_succeeded:
            verdict = "ANOMALY"  # guardrail "on" did worse than baseline — investigate
        else:
            verdict = "ANOMALY"  # baseline itself was defended — baseline should have no guard

        rows.append({
            "id": aid,
            "category": g["category"],
            "severity": g["severity"],
            "expected_hook": g["expected_hook"],
            "verdict": verdict,
            "baseline_succeeded": b_succeeded,
            "guarded_succeeded": g_succeeded,
            "caught_by_hook": g["caught_by_hook"],
            "caught_by_check": g.get("caught_by_check", "none"),
        })

    total = len(rows)
    gaps = sum(1 for r in rows if r["verdict"] == "GAP")
    defended = sum(1 for r in rows if r["verdict"] == "DEFENDED")
    anomalies = sum(1 for r in rows if r["verdict"] == "ANOMALY")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_file": baseline_path,
        "guarded_file": guarded_path,
        "summary": {
            "total": total,
            "gaps": gaps,
            "defended": defended,
            "anomalies": anomalies,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Diff a baseline run against a guarded run.")
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--guarded", default="results/guarded.json")
    parser.add_argument("--out", default="results/compare.json")
    args = parser.parse_args()

    result = compare(args.baseline, args.guarded)

    for row in result["rows"]:
        if row["verdict"] == "MISSING":
            print(f"  {row['id']:9} MISSING   ({row['note']})")
            continue
        detail = f"{row['caught_by_hook']}/{row['caught_by_check']}" if row["verdict"] == "DEFENDED" else ""
        print(f"  {row['id']:9} {row['verdict']:9} {detail}")

    s = result["summary"]
    print(f"\n{s['defended']}/{s['total']} defended, {s['gaps']} gap(s), "
          f"{s['anomalies']} anomaly/anomalies — wrote {args.out}")

    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()