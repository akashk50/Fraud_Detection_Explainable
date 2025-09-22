# scripts/engineer_features.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ----------------------------
# Paths (project-relative defaults)
# ----------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

IN_TRAIN = os.path.join(DATA_DIR, "fraud_train.parquet")
IN_VAL   = os.path.join(DATA_DIR, "fraud_val.parquet")
IN_TEST  = os.path.join(DATA_DIR, "fraud_test.parquet")

OUT_TRAIN = os.path.join(DATA_DIR, "fraud_train_fe.parquet")
OUT_VAL   = os.path.join(DATA_DIR, "fraud_val_fe.parquet")
OUT_TEST  = os.path.join(DATA_DIR, "fraud_test_fe.parquet")

# ----------------------------
# Config (tweakable)
# ----------------------------
ORIGIN_TS   = pd.Timestamp("2017-01-01")   # arbitrary anchor; only diffs matter
SHRINK_K    = 20                           # hierarchical shrinkage weight
Z_EPS       = 0.25                         # min std on log scale
COHORT_COL  = "card2"                      # shrink per-card1 toward this cohort by default
WARM_START_DEFAULT = False                 # if True, pass history into val/test (past-only, no leakage)

# ----------------------------
# Feature primitives (time-safe within a frame)
# ----------------------------
def add_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ts"] = ORIGIN_TS + pd.to_timedelta(d["TransactionDT"], unit="s")
    d["hour_of_day"] = d["ts"].dt.hour.astype("int16")
    d["day_of_week"] = d["ts"].dt.dayofweek.astype("int8")  # 0=Mon
    d["is_night"]    = d["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype("uint8")
    d["is_weekend"]  = d["day_of_week"].isin([5, 6]).astype("uint8")
    return d

def add_card_velocity_and_recency(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    prev_ts = d.groupby("card1")["ts"].shift(1)
    mins_since_prev = (d["ts"] - prev_ts).dt.total_seconds() / 60.0
    d["mins_since_prev_card1"] = mins_since_prev.fillna(1e9).astype("float32")

    d_idx = d.set_index("ts")
    g = d_idx.groupby("card1", group_keys=False)

    ct_1d  = g["TransactionID"].rolling("1D", closed="left").count()
    ct_7d  = g["TransactionID"].rolling("7D", closed="left").count()
    sum_1d = g["TransactionAmt"].rolling("1D", closed="left").sum()

    d["card1_txn_ct_1d"]  = ct_1d.astype("float32").fillna(0).to_numpy()
    d["card1_txn_ct_7d"]  = ct_7d.astype("float32").fillna(0).to_numpy()
    d["card1_amt_sum_1d"] = sum_1d.astype("float32").fillna(0).to_numpy()
    return d

def _expanding_stats(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    mean_prev = series.expanding().mean().shift(1)
    var_prev  = series.expanding().var(ddof=0).shift(1)
    std_prev  = np.sqrt(var_prev.clip(lower=0))
    return mean_prev, std_prev

def add_shrunk_amount_z(
    d: pd.DataFrame,
    cohort_col: str = COHORT_COL,
    k: int = SHRINK_K,
    eps: float = Z_EPS
) -> pd.DataFrame:
    d = d.copy()
    d["logAmt"] = np.log1p(d["TransactionAmt"]).astype("float32")
    log_sq = (d["logAmt"] ** 2).astype("float32")

    # Global expanding stats (past-only)
    glob_mean_prev, glob_std_prev = _expanding_stats(d["logAmt"])
    d["glob_mean_prev"] = glob_mean_prev
    d["glob_std_prev"]  = glob_std_prev.fillna(d["logAmt"].std())

    # Cohort stats (past-only)
    if cohort_col in d.columns:
        keyH = d[cohort_col]
        nH     = d.groupby(cohort_col).cumcount()
        sumH   = d["logAmt"].groupby(keyH).cumsum() - d["logAmt"]
        sum2H  = log_sq.groupby(keyH).cumsum() - log_sq

        d["cohort_n_prev"]    = nH.astype("float32")
        denomH = d["cohort_n_prev"].replace(0, np.nan)
        d["cohort_mean_prev"] = (sumH / denomH).astype("float32")
        varH = (sum2H / denomH) - (d["cohort_mean_prev"]**2)
        d["cohort_std_prev"]  = np.sqrt(varH.clip(lower=0)).astype("float32")

        d["cohort_mean_prev"] = d["cohort_mean_prev"].fillna(d["glob_mean_prev"])
        d["cohort_std_prev"]  = d["cohort_std_prev"].fillna(d["glob_std_prev"])
        d["cohort_n_prev"]    = d["cohort_n_prev"].fillna(0)
    else:
        d["cohort_n_prev"]    = 0
        d["cohort_mean_prev"] = d["glob_mean_prev"]
        d["cohort_std_prev"]  = d["glob_std_prev"]

    # Per-card expanding stats (past-only)
    keyC = d["card1"]
    nC    = d.groupby("card1").cumcount()
    sumC  = d["logAmt"].groupby(keyC).cumsum() - d["logAmt"]
    sum2C = log_sq.groupby(keyC).cumsum() - log_sq

    d["card_n_prev"]    = nC.astype("float32")
    denomC = d["card_n_prev"].replace(0, np.nan)
    d["card_mean_prev"] = (sumC / denomC).astype("float32")
    varC = (sum2C / denomC) - (d["card_mean_prev"]**2)
    d["card_std_prev"]  = np.sqrt(varC.clip(lower=0)).astype("float32")

    d["card_mean_prev"] = d["card_mean_prev"].fillna(d["cohort_mean_prev"])
    d["card_std_prev"]  = d["card_std_prev"].fillna(d["cohort_std_prev"])

    # Shrink per-card toward cohort
    w = (d["card_n_prev"] / (d["card_n_prev"] + k)).astype("float32")
    mu_star   = w * d["card_mean_prev"] + (1 - w) * d["cohort_mean_prev"]
    sig2_star = w * (d["card_std_prev"]**2) + (1 - w) * (d["cohort_std_prev"]**2)
    sig_star  = np.sqrt(sig2_star).clip(lower=eps)

    z_star = ((d["logAmt"] - mu_star) / sig_star).clip(-5, 5).astype("float32")

    d["card1_amt_z_shrunk"] = z_star
    d["card1_has_history"]  = (d["card_n_prev"] > 0).astype("uint8")

    d.drop(columns=[
        "glob_mean_prev","glob_std_prev",
        "cohort_n_prev","cohort_mean_prev","cohort_std_prev",
        "card_n_prev","card_mean_prev","card_std_prev","logAmt"
    ], inplace=True, errors="ignore")
    return d

def add_newness_flags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for col in ["DeviceInfo", "P_emaildomain"]:
        out_col = f"is_first_seen_{col}"
        if col in d.columns:
            s = d[col].astype("object").fillna("__missing__")
            d[out_col] = (s.groupby(s).cumcount() == 0).astype("uint8")
        else:
            d[out_col] = 0
    return d

def add_amount_rounding_flags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    cents = (d["TransactionAmt"] - np.floor(d["TransactionAmt"])).round(2)
    d["is_amt_round"] = (cents == 0).astype("uint8")
    d["is_amt_99"]    = (np.isclose(cents, 0.99)).astype("uint8")
    d["is_amt_95"]    = (np.isclose(cents, 0.95)).astype("uint8")
    return d

# ----------------------------
# Split-safe FE driver
# ----------------------------
def engineer_one_split(
    df: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    *,
    cohort_col: str = COHORT_COL,
    shrink_k: int = SHRINK_K,
    z_eps: float = Z_EPS
) -> pd.DataFrame:
    """
    Engineer features using only *past* information.
    If `history` is provided, it is concatenated BEFORE `df` solely to provide prior context;
    the returned rows are only for `df` (history rows are dropped).
    """
    if history is not None and len(history):
        work = pd.concat([history, df], axis=0, ignore_index=True)
    else:
        work = df.copy()

    # Ensure time order
    work = work.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

    # Calendar from TransactionDT
    work = add_calendar_features(work)

    # Velocity/recency per card1 (uses 'ts')
    work = add_card_velocity_and_recency(work)

    # Shrunk z-score on log amount (per card1 toward cohort)
    work = add_shrunk_amount_z(work, cohort_col=cohort_col, k=shrink_k, eps=z_eps)

    # Newness flags (first time seeing value in combined history+current)
    work = add_newness_flags(work)

    # Amount rounding patterns
    work = add_amount_rounding_flags(work)

    # Memory-friendly dtypes
    for col in ["card1_txn_ct_1d", "card1_txn_ct_7d", "card1_amt_sum_1d",
                "mins_since_prev_card1", "card1_amt_z_shrunk"]:
        if col in work.columns:
            work[col] = work[col].astype("float32")

    for col in ["hour_of_day", "day_of_week"]:
        if col in work.columns:
            work[col] = work[col].astype("int16")

    for col in ["is_night","is_weekend","card1_has_history",
                "is_first_seen_DeviceInfo","is_first_seen_P_emaildomain",
                "is_amt_round","is_amt_99","is_amt_95"]:
        if col in work.columns:
            work[col] = work[col].astype("uint8")

    # Return only the rows belonging to the current split (drop history rows)
    out = work.iloc[len(history):].copy() if (history is not None and len(history)) else work.copy()

    # Clean up helper column
    if "ts" in out.columns:
        out.drop(columns=["ts"], inplace=True)

    return out

# ----------------------------
# Public API for Streamlit / imports
# ----------------------------
def engineer_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    warm_start_val_test: bool = WARM_START_DEFAULT,
    cohort_col: str = COHORT_COL,
    shrink_k: int = SHRINK_K,
    z_eps: float = Z_EPS,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split-safe feature engineering for (train, val, test).
    Returns: (train_fe, val_fe, test_fe, artifacts)
    """
    if warm_start_val_test:
        train_fe = engineer_one_split(train_df, history=None, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)
        val_fe   = engineer_one_split(val_df,   history=train_df, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)
        hist_for_test = pd.concat([train_df, val_df], ignore_index=True)
        test_fe  = engineer_one_split(test_df,  history=hist_for_test, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)
    else:
        train_fe = engineer_one_split(train_df, history=None, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)
        val_fe   = engineer_one_split(val_df,   history=None, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)
        test_fe  = engineer_one_split(test_df,  history=None, cohort_col=cohort_col, shrink_k=shrink_k, z_eps=z_eps)

    artifacts = {
        "config": {
            "warm_start_val_test": warm_start_val_test,
            "cohort_col": cohort_col,
            "shrink_k": shrink_k,
            "z_eps": z_eps,
        },
        "shapes": {
            "train_fe": tuple(train_fe.shape),
            "val_fe":   tuple(val_fe.shape),
            "test_fe":  tuple(test_fe.shape),
        },
    }

    if verbose:
        def summarize(name, d):
            fr = d["isFraud"].mean() if "isFraud" in d.columns else np.nan
            print(f"{name}: shape={d.shape}, fraud_rate={fr:.4%}, "
                  f"period=[{int(d['TransactionDT'].min())}..{int(d['TransactionDT'].max())}]")
        print("Engineered features:")
        print("  - hour_of_day, day_of_week, is_night, is_weekend")
        print("  - mins_since_prev_card1, card1_txn_ct_1d, card1_txn_ct_7d, card1_amt_sum_1d")
        print("  - card1_amt_z_shrunk (hierarchical, log-amount), card1_has_history")
        print("  - is_first_seen_DeviceInfo, is_first_seen_P_emaildomain")
        print("  - is_amt_round, is_amt_99, is_amt_95")
        summarize("TRAIN_FE", train_fe); summarize("VAL_FE", val_fe); summarize("TEST_FE", test_fe)

    return train_fe, val_fe, test_fe, artifacts

# ----------------------------
# File IO helpers
# ----------------------------
def _read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def _write_parquet(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

# ----------------------------
# CLI (zero-flag friendly)
# ----------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Engineer split-safe features for fraud datasets.")
    parser.add_argument("--in-train", type=str, default=IN_TRAIN, help=f"Path to train parquet [default: {IN_TRAIN}]")
    parser.add_argument("--in-val",   type=str, default=IN_VAL,   help=f"Path to val   parquet [default: {IN_VAL}]")
    parser.add_argument("--in-test",  type=str, default=IN_TEST,  help=f"Path to test  parquet [default: {IN_TEST}]")
    parser.add_argument("--out-train", type=str, default=OUT_TRAIN, help=f"Output train_fe parquet [default: {OUT_TRAIN}]")
    parser.add_argument("--out-val",   type=str, default=OUT_VAL,   help=f"Output val_fe   parquet [default: {OUT_VAL}]")
    parser.add_argument("--out-test",  type=str, default=OUT_TEST,  help=f"Output test_fe  parquet [default: {OUT_TEST}]")

    parser.add_argument("--warm-start", action="store_true", help="If set, pass history into val/test (past-only).")
    parser.add_argument("--cohort-col", type=str, default=COHORT_COL)
    parser.add_argument("--shrink-k",   type=int, default=SHRINK_K)
    parser.add_argument("--z-eps",      type=float, default=Z_EPS)
    parser.add_argument("--verbose",    action="store_true")

    args = parser.parse_args()

    train = _read_parquet(args.in_train)
    val   = _read_parquet(args.in_val)
    test  = _read_parquet(args.in_test)

    train_fe, val_fe, test_fe, _ = engineer_features(
        train, val, test,
        warm_start_val_test=args.warm_start,
        cohort_col=args.cohort_col,
        shrink_k=args.shrink_k,
        z_eps=args.z_eps,
        verbose=args.verbose,
    )

    _write_parquet(train_fe, args.out_train)
    _write_parquet(val_fe,   args.out_val)
    _write_parquet(test_fe,  args.out_test)

    if args.verbose:
        print(f"Wrote: {args.out_train}")
        print(f"Wrote: {args.out_val}")
        print(f"Wrote: {args.out_test}")

if __name__ == "__main__":
    main_cli()
