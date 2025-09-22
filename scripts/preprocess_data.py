# scripts/preprocess_data.py
from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional

import numpy as np
import pandas as pd


# =============================
# Defaults / Config (overridable)
# =============================
COL_NA_THRESH_DEFAULT = 0.80
ROW_NA_THRESH_DEFAULT = 0.90

TRAIN_FRAC_DEFAULT = 0.70
VAL_FRAC_DEFAULT   = 0.15  # TEST = 1 - train - val

DELTA_K_STD_DEFAULT  = 7.0
TE_SMOOTH_K_DEFAULT  = 50
TE_MIN_COUNT_DEFAULT = 20
DROP_ORIG_AFTER_TE_DEFAULT = True

ESSENTIAL_COLS = {"TransactionID", "TransactionDT", "isFraud"}
FORCE_TE_CATEGORICAL_DEFAULT = ["addr1", "addr2"]  # coerced to categorical + forced TE

# === Project-relative defaults ===
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# RAW Kaggle CSVs (put them in data/)
TRAIN_TRANSACTION_CSV = os.path.join(DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY_CSV    = os.path.join(DATA_DIR, "train_identity.csv")

# Outputs for downstream
OUT_TRAIN = os.path.join(DATA_DIR, "fraud_train.parquet")
OUT_VAL   = os.path.join(DATA_DIR, "fraud_val.parquet")
OUT_TEST  = os.path.join(DATA_DIR, "fraud_test.parquet")  # holdout from TRAIN


# =============================
# Regex helpers
# =============================
_V_COL_RE = re.compile(r"^V\d+$")
_D_COL_RE = re.compile(r"^D\d+$")
def is_v_col(c: str) -> bool: return bool(_V_COL_RE.match(c))
def is_d_col(c: str) -> bool: return bool(_D_COL_RE.match(c))


# =============================
# Core helpers (unchanged logic)
# =============================
def chronological_split_train_val_test(
    train_full: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or (train_frac + val_frac >= 1):
        raise ValueError("TRAIN_FRAC and VAL_FRAC must satisfy 0<train<1, 0<=val<1, and train+val<1.")
    d = train_full.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    n = len(d)
    cut_tr = int(n * train_frac)
    cut_va = int(n * (train_frac + val_frac))
    train = d.iloc[:cut_tr].copy()
    val   = d.iloc[cut_tr:cut_va].copy()
    test  = d.iloc[cut_va:].copy()
    return train, val, test

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

# ---- Target encoding ----
def fit_smoothed_rate(
    train: pd.DataFrame, col: str, target: str = "isFraud",
    k: int = TE_SMOOTH_K_DEFAULT, min_count: int = TE_MIN_COUNT_DEFAULT
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

def encode_categoricals_auto(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_ohe_cardinality: int = 10,
    min_level_count: int = TE_MIN_COUNT_DEFAULT,
    drop_original: bool = True,
    dummy_dtype: str = "uint8",
    force_as_categorical: Optional[List[str]] = None,
    force_te: Optional[List[str]] = None,
    te_smooth_k: int = TE_SMOOTH_K_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    tr, va, te = train.copy(), val.copy(), test.copy()
    force_as_categorical = set(force_as_categorical or [])
    force_te = set(force_te or [])

    # Coerce specific numeric-like cats to string domain so they’re treated as categorical
    for df in (tr, va, te):
        for c in force_as_categorical:
            if c in df.columns:
                # Use pandas nullable Int64 -> string; mark missing explicitly
                try:
                    df[c] = df[c].astype("Int64").astype("string").fillna("missing")
                except Exception:
                    df[c] = df[c].astype("string").fillna("missing")

    cat_cols = tr.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    for df in (tr, va, te):
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("string").fillna("missing")

    nunique = tr[cat_cols].nunique(dropna=True) if cat_cols else pd.Series(dtype=int)
    ohe_cols = [c for c in cat_cols if (nunique.get(c, 0) <= max_ohe_cardinality)]
    te_cols  = [c for c in cat_cols if (nunique.get(c, 0) >  max_ohe_cardinality)]

    # Force certain columns into TE
    te_cols  = sorted(set(te_cols)  | (force_te & set(cat_cols)))
    ohe_cols = sorted(set(ohe_cols) - set(te_cols))

    meta: Dict = {"ohe_cols": ohe_cols, "te_cols": {}, "params": {
        "max_ohe_cardinality": max_ohe_cardinality,
        "min_level_count": min_level_count,
        "dummy_dtype": dummy_dtype,
        "drop_original": drop_original,
    }}

    # Target encode
    for col in te_cols:
        mapping, g_rate, rare = fit_smoothed_rate(tr, col, target="isFraud",
                                                  k=te_smooth_k, min_count=min_level_count)
        tr[f"{col}_rate"] = apply_smoothed_rate(tr, col, mapping, g_rate, rare)
        va[f"{col}_rate"] = apply_smoothed_rate(va, col, mapping, g_rate, rare)
        te[f"{col}_rate"] = apply_smoothed_rate(te, col, mapping, g_rate, rare)
        meta["te_cols"][col] = {
            "global_rate": g_rate,
            "rare": sorted(list(rare)),
            "keys": list(mapping.keys()),
        }
        if drop_original and col in tr.columns:
            for df in (tr, va, te):
                if col in df.columns:
                    df.drop(columns=[col], inplace=True, errors="ignore")

    # One-hot encode low-card
    if ohe_cols:
        tr_ohe = pd.get_dummies(tr, columns=ohe_cols, drop_first=False, dtype=dummy_dtype)
        va_ohe = pd.get_dummies(va, columns=ohe_cols, drop_first=False, dtype=dummy_dtype)
        te_ohe = pd.get_dummies(te, columns=ohe_cols, drop_first=False, dtype=dummy_dtype)

        # Align to TRAIN columns
        va_ohe = va_ohe.reindex(columns=tr_ohe.columns, fill_value=0)
        te_ohe = te_ohe.reindex(columns=tr_ohe.columns, fill_value=0)

        tr, va, te = tr_ohe, va_ohe, te_ohe

    return tr, va, te, meta

# ---- Imputation ----
def fit_imputation_params(train: pd.DataFrame, *, delta_k_std: float) -> Dict:
    params = {
        "categoricals_keep": set(),   # reserved hook
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
        sentinel = mu + delta_k_std * (sd if (sd is not None) else 0.0)
        sentinel = max(sentinel, float(series.max()) + 1.0)
        params["d_sentinel"][c] = sentinel
    return params

def apply_imputation(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    flag_cols = {}

    # D* block
    d_cols = [c for c in params["deltas"] if c in out.columns]
    if d_cols:
        na_mask = out[d_cols].isna()
        for c in d_cols:
            flag_cols[f"{c}__is_missing"] = na_mask[c].astype("uint8")
        d_fill_map = {c: params["d_sentinel"].get(c, 1e9) for c in d_cols}
        out[d_cols] = out[d_cols].fillna(pd.Series(d_fill_map))

    # other numerics (median) + flags where TRAIN had missing
    num_cols = [c for c in params["numerics"] if c in out.columns]
    if num_cols:
        had_miss_cols = [c for c in num_cols if c in params["had_missing"]]
        if had_miss_cols:
            na_mask = out[had_miss_cols].isna()
            for c in had_miss_cols:
                flag_cols[f"{c}__is_missing"] = na_mask[c].astype("uint8")
        num_fill_map = {c: params["medians"].get(c, -1) for c in num_cols}
        out[num_cols] = out[num_cols].fillna(pd.Series(num_fill_map))

    if flag_cols:
        flags_df = pd.DataFrame(flag_cols, index=out.index)
        out = pd.concat([out, flags_df], axis=1)

    return out.copy()


# =============================
# Public API: preprocess()
# =============================
def preprocess(
    *,
    train_transaction: Optional[pd.DataFrame] = None,
    train_identity: Optional[pd.DataFrame] = None,
    merged_train:   Optional[pd.DataFrame] = None,
    # knobs
    col_na_thresh: float = COL_NA_THRESH_DEFAULT,
    row_na_thresh: float = ROW_NA_THRESH_DEFAULT,
    train_frac: float = TRAIN_FRAC_DEFAULT,
    val_frac: float = VAL_FRAC_DEFAULT,
    te_min_count: int = TE_MIN_COUNT_DEFAULT,
    te_smooth_k: int = TE_SMOOTH_K_DEFAULT,
    drop_original_after_te: bool = DROP_ORIG_AFTER_TE_DEFAULT,
    force_te_categorical: Optional[List[str]] = None,
    delta_k_std: float = DELTA_K_STD_DEFAULT,
    max_ohe_cardinality: int = 10,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    End-to-end preprocessing:
      - Merge identity/transaction (if merged_train not supplied)
      - Chronological split into (train, val, test)
      - Drop V* block
      - Column/row pruning by missingness
      - Target-encode high-card categoricals (fit on TRAIN), OHE low-card
      - Impute numerics + delta sentinels + missing flags

    Returns:
      train_proc, val_proc, test_proc, artifacts (dict)
    """
    if merged_train is None:
        if train_transaction is None or train_identity is None:
            raise ValueError("Provide either merged_train OR both train_transaction and train_identity DataFrames.")
        df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    else:
        df = merged_train.copy()

    # basic schema checks
    for req in ("TransactionID", "TransactionDT", "isFraud"):
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' missing from input data.")

    # Ensure stable numeric type for TransactionID to avoid pyarrow dtype confusion on write
    # (object/string IDs are fine too; pick one. We'll keep it as int64 if possible.)
    try:
        df["TransactionID"] = pd.to_numeric(df["TransactionID"], errors="raise").astype("int64")
    except Exception:
        # fallback: keep as string if mixed
        df["TransactionID"] = df["TransactionID"].astype("string")

    # Split
    train_df, val_df, test_df = chronological_split_train_val_test(df, train_frac, val_frac)

    # Drop V* everywhere
    train_df = drop_v_block(train_df)
    val_df   = drop_v_block(val_df)
    test_df  = drop_v_block(test_df)

    # Decide column drops on TRAIN only; apply to all splits
    drop_cols = cols_to_drop_by_missing(train_df, col_na_thresh)
    if drop_cols:
        for d in (train_df, val_df, test_df):
            d.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Optional row filtering (very empty rows)
    train_df = train_df[row_filter_mask(train_df, row_na_thresh)].reset_index(drop=True)
    val_df   = val_df[row_filter_mask(val_df, row_na_thresh)].reset_index(drop=True)
    test_df  = test_df[row_filter_mask(test_df, row_na_thresh)].reset_index(drop=True)

    # Categorical encoding (fit on TRAIN, apply to VAL/TEST)
    force_list = list(force_te_categorical or FORCE_TE_CATEGORICAL_DEFAULT)
    train_enc, val_enc, test_enc, enc_meta = encode_categoricals_auto(
        train_df, val_df, test_df,
        max_ohe_cardinality=max_ohe_cardinality,
        min_level_count=te_min_count,
        drop_original=drop_original_after_te,
        dummy_dtype="uint8",
        force_as_categorical=force_list,
        force_te=force_list,
        te_smooth_k=te_smooth_k,
    )

    # Imputation params (fit on TRAIN), apply to all
    impute_params = fit_imputation_params(train_enc, delta_k_std=delta_k_std)
    train_proc = apply_imputation(train_enc, impute_params)
    val_proc   = apply_imputation(val_enc,   impute_params)
    test_proc  = apply_imputation(test_enc,  impute_params)

    artifacts = {
        "drop_cols": drop_cols,
        "enc_meta": enc_meta,
        "impute_params": impute_params,
        "fraud_rates": {
            "train": float(train_proc["isFraud"].mean() if "isFraud" in train_proc.columns else np.nan),
            "val":   float(val_proc["isFraud"].mean() if "isFraud" in val_proc.columns else np.nan),
            "test":  float(test_proc["isFraud"].mean() if "isFraud" in test_proc.columns else np.nan),
        },
        "shapes": {
            "train": tuple(train_proc.shape),
            "val":   tuple(val_proc.shape),
            "test":  tuple(test_proc.shape),
        },
    }

    if verbose:
        print(f"Dropped columns (> {int(col_na_thresh*100)}% NA): {len(drop_cols)}")
        if drop_cols:
            print("  Examples:", drop_cols[:10])
        te_done  = sorted(list((enc_meta.get("te_cols") or {}).keys()))
        ohe_done = sorted(list(enc_meta.get("ohe_cols") or []))
        created  = [f"{c}_rate" for c in te_done if f"{c}_rate" in train_proc.columns]
        print("Target-encoded (fit on TRAIN):", te_done)
        print("Created columns:", created)
        print("One-hot encoded (fit on TRAIN):", ohe_done)
        print("TRAIN:", artifacts["shapes"]["train"], "fraud_rate=", f"{artifacts['fraud_rates']['train']:.4%}")
        print("VAL:  ", artifacts["shapes"]["val"],   "fraud_rate=", f"{artifacts['fraud_rates']['val']:.4%}")
        print("TEST: ", artifacts["shapes"]["test"],  "fraud_rate=", f"{artifacts['fraud_rates']['test']:.4%}")

    return train_proc, val_proc, test_proc, artifacts


# =============================
# Convenience: file readers/writers
# =============================
def _read_any(path: str | Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    elif p.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type for: {path}")

def _write_parquet(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# =============================
# CLI entry (optional)
# =============================
def main_cli():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Preprocess IEEE-CIS fraud TRAIN → (train/val/test) with FE/TE/imputation."
    )
    parser.add_argument("--train-transaction", type=str, default=TRAIN_TRANSACTION_CSV,
                        help=f"Path to train_transaction.(csv|parquet) [default: {TRAIN_TRANSACTION_CSV}]")
    parser.add_argument("--train-identity",   type=str, default=TRAIN_IDENTITY_CSV,
                        help=f"Path to train_identity.(csv|parquet) [default: {TRAIN_IDENTITY_CSV}]")
    parser.add_argument("--out-dir",          type=str, default=DATA_DIR,
                        help=f"Directory to write outputs [default: {DATA_DIR}]")

    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC_DEFAULT)
    parser.add_argument("--val-frac",   type=float, default=VAL_FRAC_DEFAULT)
    parser.add_argument("--col-na-thresh", type=float, default=COL_NA_THRESH_DEFAULT)
    parser.add_argument("--row-na-thresh", type=float, default=ROW_NA_THRESH_DEFAULT)
    parser.add_argument("--te-min-count",   type=int,   default=TE_MIN_COUNT_DEFAULT)
    parser.add_argument("--te-smooth-k",    type=int,   default=TE_SMOOTH_K_DEFAULT)
    parser.add_argument("--delta-k-std",    type=float, default=DELTA_K_STD_DEFAULT)
    parser.add_argument("--max-ohe-cardinality", type=int, default=10)
    parser.add_argument("--keep-original-after-te", action="store_true",
                        help="If set, DO NOT drop original high-card categorical columns after TE.")
    parser.add_argument("--verbose", action="store_true")

    # NOTE: no required=True — defaults make it runnable with zero flags
    args = parser.parse_args()

    # Resolve inputs (default to project-relative files)
    tt_path = args.train_transaction
    ti_path = args.train_identity
    out_dir = Path(args.out_dir)

    if not os.path.exists(tt_path):
        raise FileNotFoundError(f"train_transaction file not found: {tt_path}")
    if not os.path.exists(ti_path):
        raise FileNotFoundError(f"train_identity file not found: {ti_path}")

    tt = _read_any(tt_path)
    ti = _read_any(ti_path)

    tr, va, te, art = preprocess(
        train_transaction=tt,
        train_identity=ti,
        merged_train=None,
        col_na_thresh=args.col_na_thresh,
        row_na_thresh=args.row_na_thresh,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        te_min_count=args.te_min_count,
        te_smooth_k=args.te_smooth_k,
        drop_original_after_te=not args.keep_original_after_te,
        force_te_categorical=FORCE_TE_CATEGORICAL_DEFAULT,
        delta_k_std=args.delta_k_std,
        max_ohe_cardinality=args.max_ohe_cardinality,
        verbose=args.verbose,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Use your canonical output filenames inside out_dir
    tr_out = out_dir / Path(OUT_TRAIN).name
    va_out = out_dir / Path(OUT_VAL).name
    te_out = out_dir / Path(OUT_TEST).name

    _write_parquet(tr, tr_out)
    _write_parquet(va, va_out)
    _write_parquet(te, te_out)

    if args.verbose:
        print(f"Wrote: {tr_out}")
        print(f"Wrote: {va_out}")
        print(f"Wrote: {te_out}")

if __name__ == "__main__":
    main_cli()
