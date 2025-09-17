# app.py
import os, sys, math
from collections import defaultdict

import pandas as pd
import streamlit as st

# Ensure we can import from ./scripts
THIS_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(THIS_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from decisionTree import run_miner  # uses your helperFuncs under scripts/

st.set_page_config(page_title="Fraud Rule Miner (MVP)", layout="wide")
st.title("Fraud Rule Miner (MVP)")
st.caption("Upload a CSV/Parquet → mine rules → evaluate on validation → render SQL.")

# -----------------------
# One-hot collapse config
# -----------------------
GROUP_MAP = {
    "ProductCD_": "ProductCD",
    "card6_": "card6",
    "card4_": "card4",
}
GROUP_LEVELS = {
    "ProductCD_": ["W","R","H","S","C"],
    "card6_": ["credit","debit","charge","debit_or_credit"],
    "card4_": ["visa","mastercard","discover","american_express"],
}

def _sql_ident(name: str) -> str:
    return name

def _sql_val(v):
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return str(int(v)) if float(v).is_integer() else str(v)
    return f"'{str(v)}'"

def _prefix_for_feature(feat: str):
    for pref in GROUP_MAP.keys():
        if feat.startswith(pref):
            return pref
    return None

def _extract_cat_value(prefix: str, feat: str) -> str:
    return feat[len(prefix):]

def _range_to_sql(feature, low, low_incl, high, high_incl):
    parts = []
    if low is not None:
        parts.append(f"{_sql_ident(feature)} {'>=' if low_incl else '>'} {_sql_val(low)}")
    if high is not None:
        parts.append(f"{_sql_ident(feature)} {'<=' if high_incl else '<'} {_sql_val(high)}")
    return " AND ".join(parts) if parts else "1=1"

def clauses_to_sql_where_pretty(clauses: list, group_map=GROUP_MAP, group_levels=GROUP_LEVELS):
    """Collapse one-hots back to categoricals and emit SQL + English."""
    pos_vals = defaultdict(set)
    neg_vals = defaultdict(set)
    non_cat_preds = []

    for c in clauses:
        t = c.get("type"); feat = c.get("feature"); pref = _prefix_for_feature(feat)
        if pref is not None and t == "eq":
            orig_col = group_map[pref]
            val = _extract_cat_value(pref, feat)
            (pos_vals if c.get("value") == 1 else neg_vals)[orig_col].add(val)
            continue
        if t == "eq":
            non_cat_preds.append(f"{_sql_ident(feat)} = {_sql_val(c['value'])}")
        elif t == "range":
            non_cat_preds.append(_range_to_sql(
                feat, c.get("low"), c.get("low_inclusive", False),
                c.get("high"), c.get("high_inclusive", False)
            ))

    cat_preds, cat_english = [], []
    all_cat_cols = set(pos_vals.keys()) | set(neg_vals.keys())
    for col in sorted(all_cat_cols):
        positives = sorted(pos_vals[col]); negatives = sorted(neg_vals[col])
        if positives:
            if len(positives) == 1:
                cat_preds.append(f"{_sql_ident(col)} = {_sql_val(positives[0])}")
                cat_english.append(f"{col} = {positives[0]}")
            else:
                in_list = ", ".join(_sql_val(v) for v in positives)
                cat_preds.append(f"{_sql_ident(col)} IN ({in_list})")
                cat_english.append(f"{col} in ({', '.join(positives)})")
        else:
            # only negatives; try NOT IN with full levels
            pref = next((p for p, v in GROUP_MAP.items() if v == col), None)
            levels = GROUP_LEVELS.get(pref)
            if levels:
                keep = [lvl for lvl in levels if lvl not in negatives]
                if len(keep) == 1:
                    cat_preds.append(f"{_sql_ident(col)} = {_sql_val(keep[0])}")
                    cat_english.append(f"{col} = {keep[0]}")
                else:
                    in_list = ", ".join(_sql_val(v) for v in keep)
                    cat_preds.append(f"{_sql_ident(col)} IN ({in_list})")
                    cat_english.append(f"{col} in ({', '.join(keep)})")
            else:
                notin = ", ".join(_sql_val(v) for v in negatives)
                cat_preds.append(f"{_sql_ident(col)} NOT IN ({notin})")
                cat_english.append(f"{col} not in ({', '.join(negatives)})")

    preds = [p for p in (cat_preds + non_cat_preds) if p and p != "1=1"]
    sql_where = "(" + " AND ".join(preds) + ")" if preds else "(1=1)"

    english_bits = list(cat_english)
    for c in clauses:
        if c.get("type") == "range":
            f = c["feature"]; lo, hi = c.get("low"), c.get("high")
            li, hi_incl = c.get("low_inclusive", False), c.get("high_inclusive", True)
            if lo is not None and hi is None:
                english_bits.append(f"{f} {'≥' if li else '>'} {lo}")
            elif hi is not None and lo is None:
                english_bits.append(f"{f} {'≤' if hi_incl else '<'} {hi}")
            elif lo is not None and hi is not None:
                english_bits.append(f"{f} between {lo} and {hi}")
        elif c.get("type") == "eq" and _prefix_for_feature(c["feature"]) is None:
            english_bits.append(f"{c['feature']} = {c['value']}")

    english = " AND ".join(english_bits) if english_bits else "All transactions"
    return sql_where, english

# -----------------------
# UI: upload & run
# -----------------------
uploaded = st.file_uploader("Upload CSV or Parquet with raw transactions", type=["csv","parquet"])

with st.sidebar:
    st.header("Mining parameters")
    n_bootstraps   = st.number_input("Bootstraps", 1, 500, 100, step=10)
    max_depth      = st.selectbox("Tree max_depth", [2,3], index=1)
    min_leaf       = st.number_input("min_samples_leaf", 1, 200, 5, step=1)
    snap_step      = st.number_input("Amount snap step ($)", 1, 100, 10, step=1)
    min_votes_post = st.number_input("Postprocess: min votes", 0, 100, 5, step=1)

    st.header("Catalog filters")
    min_lift    = st.slider("Min lift vs baseline", 1.0, 10.0, 3.0, 0.1)
    min_support = st.number_input("Min support (val)", 0, 100000, 50, step=10)

if uploaded is None:
    st.info("Upload a file to start. Tip: you can drop in your preprocessed parquet or a CSV.")
    st.stop()

# Read uploaded file
if uploaded.name.endswith(".csv"):
    df_in = pd.read_csv(uploaded)
else:
    df_in = pd.read_parquet(uploaded)

st.write("### Data preview")
st.dataframe(df_in.head(), use_container_width=True)

with st.spinner("Mining rules…"):
    rule_eval, cleaned_rules, baseline_val = run_miner(
        df_in,
        n_bootstraps=n_bootstraps,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        snap_step=snap_step,
        min_votes_post=min_votes_post
    )

# Compute lift and prettify SQL/English per rule
rule_eval = rule_eval.copy()
rule_eval["lift"] = rule_eval["precision"] / baseline_val

sql_list = []; english_list = []
for _, row in rule_eval.iterrows():
    sql_where, english = clauses_to_sql_where_pretty(row["rule"])
    sql_list.append(sql_where); english_list.append(english)
rule_eval["sql_where"] = sql_list
rule_eval["english"]   = english_list

st.success(f"✅ Done. Baseline fraud rate on validation: {baseline_val:.4%}")

# Filter for display
mask = (rule_eval["lift"] >= min_lift) & (rule_eval["support"] >= min_support)
filtered = rule_eval.loc[mask].sort_values(["lift","precision","support"], ascending=False)

st.subheader("Rule catalog (filtered)")
st.dataframe(
    filtered[["sql_where","english","precision","lift","support","votes"]].reset_index(drop=True),
    use_container_width=True, height=420
)

# Details + export
if not filtered.empty:
    st.subheader("Details")
    idx = st.number_input("Show rule #", 0, len(filtered)-1, 0, step=1)
    row = filtered.iloc[int(idx)]
    st.code(row["sql_where"], language="sql")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{row['precision']:.2%}")
    c2.metric("Lift", f"{row['lift']:.2f}")
    c3.metric("Support", f"{int(row['support'])}")
    c4.metric("Votes", f"{int(row['votes'])}" if "votes" in row else "—")
    with st.expander("Raw clauses"):
        st.write(row["rule"])

st.subheader("Export")
out_cols = ["sql_where","english","precision","lift","support","votes","rule"]
out_df = filtered[[c for c in out_cols if c in filtered.columns]].reset_index(drop=True)
st.download_button(
    "Download filtered catalog (CSV)",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="rule_catalog_filtered.csv",
    mime="text/csv"
)
