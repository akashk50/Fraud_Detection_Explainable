# scripts/engineer_features.py
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Paths
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
# Config
# ----------------------------
ORIGIN_TS  = pd.Timestamp("2017-01-01")  # arbitrary; only diffs matter

# Shrunk z-score params
SHRINK_K = 20      # weight prior strength
Z_EPS    = 0.25    # minimum std on log scale
COHORT_COL = "card2"  # shrink per-card1 toward this cohort by default

# If True, pass *train* as "history" when computing val/test features (past-only, no future leakage)
WARM_START_VAL_TEST = False


# ----------------------------
# Feature primitives (time-safe within a frame)
# ----------------------------
def add_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    d["ts"] = ORIGIN_TS + pd.to_timedelta(d["TransactionDT"], unit="s")
    d["hour_of_day"] = d["ts"].dt.hour.astype("int16")
    d["day_of_week"] = d["ts"].dt.dayofweek.astype("int8")  # 0=Mon
    d["is_night"]    = d["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype("uint8")
    d["is_weekend"]  = d["day_of_week"].isin([5, 6]).astype("uint8")
    return d


def add_card_velocity_and_recency(d: pd.DataFrame) -> pd.DataFrame:
    # previous timestamp per card
    prev_ts = d.groupby("card1")["ts"].shift(1)

    # minutes since previous txn; large sentinel if no history
    mins_since_prev = (d["ts"] - prev_ts).dt.total_seconds() / 60.0
    d["mins_since_prev_card1"] = mins_since_prev.fillna(1e9).astype("float32")

    # time-windowed counts/sums using DatetimeIndex; exclude current row with closed="left"
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
    # returns (mean_prev, std_prev) using *past only* (shifted expanding)
    mean_prev = series.expanding().mean().shift(1)
    var_prev  = series.expanding().var(ddof=0).shift(1)
    std_prev  = np.sqrt(var_prev.clip(lower=0))
    return mean_prev, std_prev


def add_shrunk_amount_z(d: pd.DataFrame,
                        cohort_col: str = COHORT_COL,
                        k: int = SHRINK_K,
                        eps: float = Z_EPS) -> pd.DataFrame:
    d["logAmt"] = np.log1p(d["TransactionAmt"]).astype("float32")
    log_sq = (d["logAmt"] ** 2).astype("float32")

    # ---- Global expanding stats (past-only)
    glob_mean_prev, glob_std_prev = _expanding_stats(d["logAmt"])
    d["glob_mean_prev"] = glob_mean_prev
    d["glob_std_prev"]  = glob_std_prev.fillna(d["logAmt"].std())

    # ---- Cohort stats (past-only)
    if cohort_col in d.columns:
        keyH = d[cohort_col]
        nH     = d.groupby(cohort_col).cumcount()                                # past count in cohort
        sumH   = d["logAmt"].groupby(keyH).cumsum() - d["logAmt"]                # past sum
        sum2H  = log_sq.groupby(keyH).cumsum() - log_sq                          # past sum of squares

        d["cohort_n_prev"]    = nH.astype("float32")
        denomH = d["cohort_n_prev"].replace(0, np.nan)
        d["cohort_mean_prev"] = (sumH / denomH).astype("float32")
        varH = (sum2H / denomH) - (d["cohort_mean_prev"]**2)
        d["cohort_std_prev"]  = np.sqrt(varH.clip(lower=0)).astype("float32")

        # fill with global where no cohort history
        d["cohort_mean_prev"] = d["cohort_mean_prev"].fillna(d["glob_mean_prev"])
        d["cohort_std_prev"]  = d["cohort_std_prev"].fillna(d["glob_std_prev"])
        d["cohort_n_prev"]    = d["cohort_n_prev"].fillna(0)
    else:
        d["cohort_n_prev"]    = 0
        d["cohort_mean_prev"] = d["glob_mean_prev"]
        d["cohort_std_prev"]  = d["glob_std_prev"]

    # ---- Per-card expanding stats (past-only)
    keyC = d["card1"]
    nC    = d.groupby("card1").cumcount()
    sumC  = d["logAmt"].groupby(keyC).cumsum() - d["logAmt"]
    sum2C = log_sq.groupby(keyC).cumsum() - log_sq

    d["card_n_prev"]    = nC.astype("float32")
    denomC = d["card_n_prev"].replace(0, np.nan)
    d["card_mean_prev"] = (sumC / denomC).astype("float32")
    varC = (sum2C / denomC) - (d["card_mean_prev"]**2)
    d["card_std_prev"]  = np.sqrt(varC.clip(lower=0)).astype("float32")

    # Fill first-observation NaNs with cohort/global
    d["card_mean_prev"] = d["card_mean_prev"].fillna(d["cohort_mean_prev"])
    d["card_std_prev"]  = d["card_std_prev"].fillna(d["cohort_std_prev"])

    # ---- Shrinkage toward cohort
    w = (d["card_n_prev"] / (d["card_n_prev"] + k)).astype("float32")
    mu_star   = w * d["card_mean_prev"] + (1 - w) * d["cohort_mean_prev"]
    sig2_star = w * (d["card_std_prev"]**2) + (1 - w) * (d["cohort_std_prev"]**2)
    sig_star  = np.sqrt(sig2_star).clip(lower=eps)

    z_star = ((d["logAmt"] - mu_star) / sig_star).clip(-5, 5).astype("float32")

    d["card1_amt_z_shrunk"] = z_star
    d["card1_has_history"]  = (d["card_n_prev"] > 0).astype("uint8")

    # drop helpers
    d.drop(columns=["glob_mean_prev","glob_std_prev",
                    "cohort_n_prev","cohort_mean_prev","cohort_std_prev",
                    "card_n_prev","card_mean_prev","card_std_prev","logAmt"],
           inplace=True, errors="ignore")
    return d


def add_newness_flags(d: pd.DataFrame) -> pd.DataFrame:
    for col in ["DeviceInfo", "P_emaildomain"]:
        if col in d.columns:
            s = d[col].astype("object").fillna("__missing__")
            d[f"is_first_seen_{col}"] = (s.groupby(s).cumcount() == 0).astype("uint8")
        else:
            d[f"is_first_seen_{col}"] = 0
    return d


def add_amount_rounding_flags(d: pd.DataFrame) -> pd.DataFrame:
    cents = (d["TransactionAmt"] - np.floor(d["TransactionAmt"])).round(2)
    d["is_amt_round"] = (cents == 0).astype("uint8")
    d["is_amt_99"]    = (np.isclose(cents, 0.99)).astype("uint8")
    d["is_amt_95"]    = (np.isclose(cents, 0.95)).astype("uint8")
    return d


# ----------------------------
# Split-safe FE driver
# ----------------------------
def engineer_one_split(df: pd.DataFrame,
                       history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Engineer features using only *past* information.
    If `history` is provided, it is concatenated BEFORE `df` solely to provide prior context;
    the returned rows are only for `df` (history rows are dropped).
    """
    # Build working frame: history (optional) + current split
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
    work = add_shrunk_amount_z(work, cohort_col=COHORT_COL, k=SHRINK_K, eps=Z_EPS)

    # Newness flags (first time seeing value in the *combined* history+current)
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
    if history is not None and len(history):
        out = work.iloc[len(history):].copy()
    else:
        out = work

    # Clean up helper column
    if "ts" in out.columns:
        out.drop(columns=["ts"], inplace=True)

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train = pd.read_parquet(IN_TRAIN)
    val   = pd.read_parquet(IN_VAL)
    test  = pd.read_parquet(IN_TEST)

    # Strict isolation by default; flip WARM_START_VAL_TEST=True to let val/test see train's *past* only
    if WARM_START_VAL_TEST:
        train_fe = engineer_one_split(train, history=None)
        val_fe   = engineer_one_split(val,   history=train)
        test_fe  = engineer_one_split(test,  history=pd.concat([train, val], ignore_index=True))
    else:
        train_fe = engineer_one_split(train, history=None)
        val_fe   = engineer_one_split(val,   history=None)
        test_fe  = engineer_one_split(test,  history=None)

    # Save
    train_fe.to_parquet(OUT_TRAIN, index=False)
    val_fe.to_parquet(OUT_VAL, index=False)
    test_fe.to_parquet(OUT_TEST, index=False)

    # Quick summary
    def summarize(name, d):
        print(
            f"{name}: shape={d.shape}, "
            f"fraud_rate={d['isFraud'].mean():.4f}, "
            f"period=[{int(d['TransactionDT'].min())}..{int(d['TransactionDT'].max())}]"
        )

    print("Engineered features added:")
    print("  - hour_of_day, day_of_week, is_night, is_weekend")
    print("  - mins_since_prev_card1, card1_txn_ct_1d, card1_txn_ct_7d, card1_amt_sum_1d")
    print("  - card1_amt_z_shrunk (hierarchical, log-amount), card1_has_history")
    print("  - is_first_seen_DeviceInfo, is_first_seen_P_emaildomain")
    print("  - is_amt_round, is_amt_99, is_amt_95")
    summarize("TRAIN", train_fe)
    summarize("VAL",   val_fe)
    summarize("TEST",  test_fe)


if __name__ == "__main__":
    main()
