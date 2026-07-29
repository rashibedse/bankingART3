"""
Streamlit dashboard — pure presentation layer.

Reads the three JSON files the pipeline already produces:
    results/baseline.json      (runner/run.py --guardrail off)
    results/guarded.json       (runner/run.py --guardrail on)
    results/compare.json       (runner/compare.py)

Does no computation of its own — every number here already exists in one
of those files. This file only decides how to show it.

Run from the project root:
    pip install streamlit pandas
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agent Guardrail Red-Team Dashboard", layout="wide")


def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


baseline = load_json("results/baseline_llmweak.json")
guarded = load_json("results/guardrail_llmweak.json")
compare = load_json("results/compareweak.json")

st.title("Agent Guardrail Red-Team Dashboard")

missing = [name for name, data in
           [("results/baseline_llmweak.json", baseline), ("results/guardrail_llmweak.json", guarded), ("results/compareweak.json", compare)]
           if data is None]
if missing:
    st.warning(
        "Missing: " + ", ".join(missing) +
        ".\n\nRun, in order:\n"
        "```\npython runner/run.py --guardrail off\n"
        "python runner/run.py --guardrail on\n"
        "python runner/compare.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------- summary
s = compare["summary"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total attacks", s["total"])
col2.metric("Defended", s["defended"])
col3.metric("Gaps (still succeed)", s["gaps"])
col4.metric("Anomalies", s["anomalies"])
st.caption(
    f"Baseline: {baseline['generated_at']}  ·  Guarded: {guarded['generated_at']}"
)

# --------------------------------------------------- before/after table
st.subheader("Before / after — every attack")
df = pd.DataFrame(compare["rows"])
df = df[df["verdict"] != "MISSING"]  # MISSING rows have a different shape


def _highlight(row):
    color = {
        "GAP": "background-color: #ffd6d6",
        "DEFENDED": "background-color: #d6ffd9",
        "ANOMALY": "background-color: #fff3b0",
    }.get(row["verdict"], "")
    return [color] * len(row)


display_cols = ["id", "category", "severity", "expected_hook", "verdict",
                 "caught_by_hook", "caught_by_check"]
st.dataframe(
    df[display_cols].style.apply(_highlight, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------- gap callout
gap_rows = df[df["verdict"] == "GAP"]
if len(gap_rows):
    st.subheader("⚠️ Known gaps — attacks the guardrail does not stop")
    for _, r in gap_rows.iterrows():
        st.error(
            f"**{r['id']}** ({r['category']}, severity: {r['severity']}) — "
            f"succeeded in both the baseline and guarded runs. "
            f"Expected hook: `{r['expected_hook']}`. "
            "This is a deterministic-filter blind spot, not a run failure — "
            "it's the argument for adding a model-based judge layer."
        )
else:
    st.success("No gaps — every attack that succeeded in baseline was stopped when guarded.")

# ------------------------------------------------------------ breakdown
st.subheader("Which layer is catching attacks")
defended_df = df[df["verdict"] == "DEFENDED"]
if len(defended_df):
    counts = defended_df["caught_by_check"].value_counts()
    st.bar_chart(counts)
    st.caption(
        "Each bar is a distinct check in the stack. A single dominant bar "
        "would mean the stack isn't earning its keep — spread across "
        "several checks means defense-in-depth is actually doing work."
    )
else:
    st.info("No defended attacks yet to break down.")

# -------------------------------------------------------------- drill-down
st.subheader("Attack trace drill-down")
mode = st.radio("View run:", ["guarded", "baseline"], horizontal=True)
data = guarded if mode == "guarded" else baseline
ids = [r["id"] for r in data["results"]]
selected = st.selectbox("Select attack", ids)
record = next(r for r in data["results"] if r["id"] == selected)

colA, colB = st.columns(2)
with colA:
    st.markdown(
        f"**Category:** {record['category']}  \n"
        f"**Severity:** {record['severity']}  \n"
        f"**Expected hook:** {record['expected_hook']}"
    )
with colB:
    outcome = "SUCCEEDED" if record["succeeded"] else f"defended ({record['caught_by_hook']})"
    st.markdown(
        f"**Result:** {outcome}  \n"
        f"**Success check:** {record['success_check']}  \n"
        f"**Caught by check:** {record.get('caught_by_check', 'none')}"
    )

st.markdown("**Poison planted:**")
st.code(record["trace"]["poison_planted"])

st.markdown(f"**Before-guard verdict:** `{record['trace']['before_verdict']}`")
before_checks = record["trace"].get("before_checks", [])
if before_checks:
    st.table(pd.DataFrame(before_checks))

st.markdown("**Agent's decided tool calls:**")
st.json(record["trace"]["agent_tool_calls"])

st.markdown(f"**After-guard verdict:** `{record['trace']['after_verdict']}`")
after_checks = record["trace"].get("after_checks", [])
if after_checks:
    st.table(pd.DataFrame(after_checks))

st.markdown("**Output inspected by after-guard:**")
st.code(record["trace"]["output_inspected"] or "(none — blocked before agent acted)")