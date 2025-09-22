# app.py (relevant diffs only)
import os, sys, math
from collections import defaultdict
from pathlib import Path
import pandas as pd
import streamlit as st

THIS_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(THIS_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from scripts.preprocess_data import preprocess
from scripts.feature_engineering import engineer_features
from scripts.feature_select import select_features
from scripts.decisionTree import run_miner_from_frames  # ← use this

st.set_page_config(page_title="Fraud Rule Miner (MVP)", layout="wide")
st.title("Fraud Rule Miner (MVP)")
st.caption("Upload raw IEEE-CIS tables or a merged file → preprocess → FE → feature select → mine rules → SQL.")

# -----------------------
# Sidebar: pipeline params
# -----------------------
with st.sidebar:
    st.header("Input mode")
    input_mode = st.radio("Choose input", ["Raw IEEE-CIS (identity + transaction)", "Single merged file"], index=0)

    st.header("Preprocess")
    valid_size = st.slider("Validation size", 0.05, 0.5, 0.2, 0.05)
    test_size  = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
    seed       = st.number_input("Random seed", 0, 1_000_000, 42, step=1)

    st.header("Feature Engineering")
    # add FE params as needed, e.g. thresholds, encoders, etc.
    fe_params = {}

    st.header("Feature Selection")
    # add FS params, e.g. top_k, rfe_n_features
    fs_params = {}

    st.header("Rule Mining")
    n_bootstraps   = st.number_input("Bootstraps", 1, 500, 100, step=10)
    max_depth      = st.selectbox("Tree max_depth", [2,3], index=1)
    min_leaf       = st.number_input("min_samples_leaf", 1, 200, 5, step=1)
    snap_step      = st.number_input("Amount snap step ($)", 1, 100, 10, step=1)
    min_votes_post = st.number_input("Postprocess: min votes", 0, 100, 5, step=1)

    st.header("Catalog filters")
    min_lift    = st.slider("Min lift vs baseline", 1.0, 10.0, 3.0, 0.1)
    min_support = st.number_input("Min support (val)", 0, 100000, 50, step=10)

# -----------------------
# Uploads
# -----------------------
if input_mode == "Raw IEEE-CIS (identity + transaction)":
    up_id = st.file_uploader("Upload train_identity (CSV/Parquet)", type=["csv","parquet"], key="id")
    up_tx = st.file_uploader("Upload train_transaction (CSV/Parquet)", type=["csv","parquet"], key="tx")
    if not up_id or not up_tx:
        st.info("Upload both tables to start.")
        st.stop()
else:
    uploaded = st.file_uploader("Upload single merged CSV/Parquet", type=["csv","parquet"], key="merged")
    if not uploaded:
        st.info("Upload a file to start.")
        st.stop()

# -----------------------
# Helpers
# -----------------------
def _read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_parquet(file)

@st.cache_data(show_spinner=False)
def run_full_pipeline_from_raw(id_bytes, tx_bytes, valid_size, test_size, seed, fe_params, fs_params):
    df_id = _read_any(id_bytes)
    df_tx = _read_any(tx_bytes)

    train_frac = max(0.0, 1.0 - float(valid_size) - float(test_size))

    tr, va, te, _ = preprocess(
        train_transaction=df_tx,
        train_identity=df_id,
        merged_train=None,
        train_frac=train_frac,
        val_frac=float(valid_size),
        verbose=False,
    )

    tr_fe, va_fe, te_fe, _ = engineer_features(tr, va, te, **fe_params)
    tr_fs, va_fs, te_fs, _ = select_features(tr_fe, va_fe, te_fe, **fs_params)
    return tr_fs, va_fs, te_fs

@st.cache_data(show_spinner=False)
def run_full_pipeline_from_merged(merged_bytes, valid_size, test_size, seed, fe_params, fs_params):
    df = _read_any(merged_bytes)

    train_frac = max(0.0, 1.0 - float(valid_size) - float(test_size))

    tr, va, te, _ = preprocess(
        merged_train=df,
        train_transaction=None,
        train_identity=None,
        train_frac=train_frac,
        val_frac=float(valid_size),
        verbose=False,
    )

    tr_fe, va_fe, te_fe, _ = engineer_features(tr, va, te, **fe_params)
    tr_fs, va_fs, te_fs, _ = select_features(tr_fe, va_fe, te_fe, **fs_params)
    return tr_fs, va_fs, te_fs


# -----------------------
# Run pipeline
# -----------------------
with st.spinner("Running pipeline…"):
    if input_mode == "Raw IEEE-CIS (identity + transaction)":
        train_df, val_df, test_df = run_full_pipeline_from_raw(up_id, up_tx, valid_size, test_size, seed, fe_params, fs_params)
    else:
        train_df, val_df, test_df = run_full_pipeline_from_merged(uploaded, valid_size, test_size, seed, fe_params, fs_params)

st.write("### Validation preview")
st.dataframe(val_df.head(), use_container_width=True)

# -----------------------
# Rule mining on validation
# -----------------------
try:
    with st.spinner("Mining rules…"):
        rule_eval, cleaned_rules, baseline_val = run_miner_from_frames(
            train_df, val_df,
            n_bootstraps=int(n_bootstraps),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_leaf),
            min_votes_post=int(min_votes_post),
            seed=int(seed),
        )
except Exception as e:
    st.error("Rule mining failed.")
    st.exception(e)
    st.stop()

# compute lift & pretty SQL/English (reuse your helpers)
# ... (your existing clauses_to_sql_where_pretty + catalog UI)

# --- After mining results (paste right after the miner try/except) ---

# 1) baseline + quick sanity
st.success(f"✅ Done. Baseline fraud rate on validation: {baseline_val:.4%}")
st.caption(f"Discovered {len(rule_eval)} rules before filtering.")

# 2) compute lift (if not already present)
if "precision" in rule_eval.columns and "lift" not in rule_eval.columns:
    rule_eval = rule_eval.copy()
    rule_eval["lift"] = rule_eval["precision"] / max(baseline_val, 1e-12)

# 3) filter + sort
mask = (rule_eval.get("lift", 0) >= float(min_lift)) & (rule_eval.get("support", 0) >= int(min_support))
filtered = rule_eval.loc[mask].sort_values(["lift","precision","support"], ascending=False)

st.subheader("Rule catalog (filtered)")
if filtered.empty:
    st.info("No rules match the current filters. Try lowering Min lift or Min support.")
else:
    st.dataframe(
        filtered[["sql_where","english","precision","lift","support","votes"]].reset_index(drop=True),
        use_container_width=True, height=420
    )

    # 4) details
    st.subheader("Details")
    idx = st.number_input("Show rule #", 0, len(filtered)-1, 0, step=1)
    row = filtered.iloc[int(idx)]

    st.code(row["sql_where"], language="sql")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{row['precision']:.2%}")
    c2.metric("Lift", f"{row['lift']:.2f}")
    c3.metric("Support", f"{int(row['support'])}")
    c4.metric("Votes",  f"{int(row['votes'])}" if "votes" in row else "—")

    with st.expander("Raw clauses"):
        st.write(row["rule"])

    # 5) export
    st.subheader("Export")
    out_cols = ["sql_where","english","precision","lift","support","votes","rule"]
    out_df = filtered[[c for c in out_cols if c in filtered.columns]].reset_index(drop=True)
    st.download_button(
        "Download filtered catalog (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="rule_catalog_filtered.csv",
        mime="text/csv"
    )
