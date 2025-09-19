# scripts/preprocess_data.py
import os
import re
from typing import Dict, Tuple, List, Set

import numpy as np
import pandas as pd

# =========================================
# Paths
# =========================================
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_PATH          = os.path.join(DATA_DIR, "train_merged.parquet")
FE_TRAIN_PATH     = os.path.join(DATA_DIR, "fraud_train_fe.parquet")
FE_VAL_PATH       = os.path.join(DATA_DIR, "fraud_val_fe.parquet")
FE_TEST_PATH      = os.path.join(DATA_DIR, "fraud_test_fe.parquet")

OUT_TRAIN = os.path.join(DATA_DIR, "fraud_train.parquet")
OUT_VAL   = os.path.join(DATA_DIR, "fraud_val.parquet")
OUT_TEST  = os.path.join(DATA_DIR, "fraud_test.parquet")

# =========================================
# Config (tweakable)
# =========================================
COL_NA_THRESH = 0.80    # drop columns with >80% missing (decide using TRAIN only)
ROW_NA_THRESH = 0.90    # drop rows with >90% missing (after column drops)
TRAIN_FRAC    = 0.70    # only used in fallback mode (no FE files)
VAL_FRAC      = 0.15

DELTA_K_STD   = 7.0     # sentinel for D* = mean + k*std (per D col on TRAIN)
TE_SMOOTH_K   = 50      # smoothed target encoding shrinkage strength
TE_MIN_COUNT  = 20      # rare categories (< min_count) bucketed into "__other__"
TE_HIGH_CARD_COLS = ["addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceInfo"]
DROP_ORIGINAL_AFTER_TE = True

LOW_CARD_CATS = ["ProductCD", "card4", "card6", "DeviceType"]
ESSENTIAL_COLS = {"TransactionID", "TransactionDT", "isFraud"}

# =========================================
# Regex helpers
# =========================================
_V_COL_RE = re.compile(r"^V\d+$")
_D_COL_RE = re.compile(r"^D\d+$")
def is_v_col(c: str) -> bool: return bool(_V_COL_RE.match(c))
def is_d_col(c: str) -> bool: return bool(_D_COL_RE.match(c))

# =========================================
# IO & split helpers
# =========================================
def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    n = len(d)
    cut_tr = int(n * TRAIN_FRAC)
    cut_va = int(n * (TRAIN_FRAC + VAL_FRAC))
    return d.iloc[:cut_tr].copy(), d.iloc[cut_tr:cut_va].copy(), d.iloc[cut_va:].copy()

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load FE splits if present; otherwise fall back to RAW and split now."""
    if all(os.path.exists(p) for p in [FE_TRAIN_PATH, FE_VAL_PATH, FE_TEST_PATH]):
        train = pd.read_parquet(FE_TRAIN_PATH)
        val   = pd.read_parquet(FE_VAL_PATH)
        test  = pd.read_parquet(FE_TEST_PATH)
        return train, val, test, True
    else:
        print("FE splits not found; falling back to RAW and performing split here.")
        df = pd.read_parquet(RAW_PATH)
        return chronological_split(df) + (False,)

# =========================================
# Column dropping & row filtering
# =========================================
def drop_v_block(df: pd.DataFrame) -> pd.DataFrame:
    v_cols = [c for c in df.columns if is_v_col(c)]
    return df.drop(columns=v_cols, errors="ignore")

def cols_to_drop_by_missing(train: pd.DataFrame, threshold: float) -> List[str]:
    na_ratio = train.isna().mean()
    return [c for c, r in na_ratio.items() if (r > threshold) and (c not in ESSENTIAL_COLS)]

def row_filter_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    feat_cols = [c for c in df.columns if c not in ESSENTIAL_COLS]
    if not feat_cols:
        return pd.Series(True, index=df.index)
    na_ratio_row = df[feat_cols].isna().mean(axis=1)
    return na_ratio_row <= threshold

# =========================================
# Target encoding (smoothed fraud rate, TRAIN-only)
# =========================================
def fit_smoothed_rate(
    train: pd.DataFrame, col: str, target: str = "isFraud",
    k: int = TE_SMOOTH_K, min_count: int = TE_MIN_COUNT
) -> Tuple[Dict[str, float], float, Set[str]]:
    s = train[col].astype("object").fillna("__missing__")
    y = train[target].astype(int)
    vc = s.value_counts()
    rare_set = set(vc[vc < min_count].index)
    s_grouped = s.where(~s.isin(rare_set), "__other__")
    global_rate = float(y.mean())
    stats = pd.DataFrame({"cat": s_grouped, "y": y}).groupby("cat")["y"].agg(["mean", "count"])
    stats["smoothed"] = (stats["count"] * stats["mean"] + k * global_rate) / (stats["count"] + k)
    mapping = stats["smoothed"].to_dict()
    mapping.setdefault("__missing__", global_rate)
    mapping.setdefault("__other__",   global_rate)
    return mapping, global_rate, rare_set

def apply_smoothed_rate(df: pd.DataFrame, col: str, mapping: Dict[str, float], global_rate: float, rare_set: Set[str]) -> pd.Series:
    s = df[col].astype("object").fillna("__missing__")
    s_grouped = s.where(~s.isin(rare_set), "__other__")
    return s_grouped.map(mapping).fillna(global_rate)

def target_encode_high_cardinals(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cols: List[str]):
    encoders = {}
    for col in cols:
        if col not in train.columns:
            continue
        mapping, g_rate, rare = fit_smoothed_rate(train, col, target="isFraud")
        train[f"{col}_rate"] = apply_smoothed_rate(train, col, mapping, g_rate, rare)
        val[f"{col}_rate"]   = apply_smoothed_rate(val,   col, mapping, g_rate, rare)
        test[f"{col}_rate"]  = apply_smoothed_rate(test,  col, mapping, g_rate, rare)
        encoders[col] = {"global_rate": g_rate, "keys": list(mapping.keys()), "min_count": TE_MIN_COUNT}
        if DROP_ORIGINAL_AFTER_TE:
            train.drop(columns=[col], inplace=True)
            val.drop(columns=[col], inplace=True)
            test.drop(columns=[col], inplace=True)
    return train, val, test, encoders

# =========================================
# Imputation params (TRAIN-only) & application
# =========================================
def fit_imputation_params(train: pd.DataFrame) -> Dict:
    params = {
        "categoricals_keep": set([c for c in LOW_CARD_CATS if c in train.columns]),
        "numerics": set(),
        "deltas": set(),
        "medians": {},
        "d_sentinel": {},
        "had_missing": set(),
    }
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    d_cols = [c for c in numeric_cols if is_d_col(c)]
    other_num = [c for c in numeric_cols if (c not in d_cols) and (c not in ESSENTIAL_COLS)]
    params["deltas"] = set(d_cols)
    params["numerics"] = set(other_num)

    na_any = train.isna().any()
    params["had_missing"] = set([c for c, m in na_any.items() if m])

    for c in other_num:
        params["medians"][c] = float(train[c].median(skipna=True)) if c in train.columns else None

    for c in d_cols:
        series = train[c].dropna()
        if len(series) == 0:
            params["d_sentinel"][c] = 1e9
            continue
        mu = float(series.mean())
        sd = float(series.std(ddof=0))
        sentinel = mu + DELTA_K_STD * (sd if sd is not None else 0.0)
        sentinel = max(sentinel, float(series.max()) + 1.0)
        params["d_sentinel"][c] = sentinel
    return params

def apply_imputation(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()

    # low-card categoricals: literal "missing"
    for c in params["categoricals_keep"]:
        if c in out.columns:
            out[c] = out[c].astype("object").fillna("missing")

    # D* with sentinel + flags
    for c in params["deltas"]:
        if c in out.columns:
            out[f"{c}__is_missing"] = out[c].isna().astype("uint8")
            out[c] = out[c].fillna(params["d_sentinel"].get(c, 1e9))

    # other numerics: flags (if had missing in TRAIN) + median fill
    for c in params["numerics"]:
        if c in out.columns:
            if c in params["had_missing"]:
                out[f"{c}__is_missing"] = out[c].isna().astype("uint8")
            med = params["medians"].get(c, None)
            out[c] = out[c].fillna(med if med is not None else -1)

    return out

# =========================================
# Main
# =========================================
def preprocess_main():
    # 0) Load splits (prefer FE splits)
    train, val, test, from_fe = load_splits()

    # 1) Drop V1..V339 (whether FE or RAW)
    train = drop_v_block(train)
    val   = drop_v_block(val)
    test  = drop_v_block(test)

    # 2) If we fell back to RAW earlier, we just split; now continue as usual

    # 3) Column drop list decided on TRAIN only
    drop_cols = cols_to_drop_by_missing(train, COL_NA_THRESH)
    if drop_cols:
        train.drop(columns=drop_cols, inplace=True, errors="ignore")
        val.drop(columns=drop_cols, inplace=True, errors="ignore")
        test.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 4) Optional row filtering (very empty rows)
    train = train[row_filter_mask(train, ROW_NA_THRESH)].reset_index(drop=True)
    val   = val[row_filter_mask(val, ROW_NA_THRESH)].reset_index(drop=True)
    test  = test[row_filter_mask(test, ROW_NA_THRESH)].reset_index(drop=True)

    # 5) Target-encode high-card categoricals (TRAIN-only fit)
    present_high = [c for c in TE_HIGH_CARD_COLS if c in train.columns]
    train, val, test, te_info = target_encode_high_cardinals(train, val, test, present_high)

    # 6) Fit imputers on TRAIN; apply to all
    params = fit_imputation_params(train)
    train_proc = apply_imputation(train, params)
    val_proc   = apply_imputation(val, params)
    test_proc  = apply_imputation(test, params)

    # 7) Save
    os.makedirs(DATA_DIR, exist_ok=True)
    train_proc.to_parquet(OUT_TRAIN, index=False)
    val_proc.to_parquet(OUT_VAL, index=False)
    test_proc.to_parquet(OUT_TEST, index=False)

    # 8) Summary
    def summarize(name: str, d: pd.DataFrame):
        rate = d["isFraud"].mean() if "isFraud" in d.columns else float("nan")
        print(f"{name}: shape={d.shape}, fraud_rate={rate:.4f}")

    print(f"Dropped columns (> {int(COL_NA_THRESH*100)}% NA): {len(drop_cols)}")
    if drop_cols: print("  Examples:", drop_cols[:10])
    if present_high:
        print("Target-encoded (train-only):", present_high)
        print("Created columns:", [f"{c}_rate" for c in present_high if f"{c}_rate" in train_proc.columns])
    print("Loaded from FE splits?" , from_fe)
    summarize("TRAIN", train_proc)
    summarize("VAL",   val_proc)
    summarize("TEST",  test_proc)

if __name__ == "__main__":
    preprocess_main()
