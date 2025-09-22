# scripts/feature_select.py
from __future__ import annotations

import os, json, argparse
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

# --------------------------------
# Project-relative default paths
# --------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "fraud_train_fe.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "fraud_val_fe.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "fraud_test_fe.parquet")

OUT_TRAIN  = os.path.join(DATA_DIR, "fraud_train_reduced.parquet")
OUT_VAL    = os.path.join(DATA_DIR, "fraud_val_reduced.parquet")
OUT_TEST   = os.path.join(DATA_DIR, "fraud_test_reduced.parquet")

OUT_LIST   = os.path.join(DATA_DIR, "feature_keep.json")
OUT_RANKS  = os.path.join(DATA_DIR, "feature_rank_shap.csv")
OUT_REPORT = os.path.join(DATA_DIR, "feature_clusters.csv")

# --------------------------------
# Config (tweakable)
# --------------------------------
AUC_DELTA_MIN = 0.01      # |AUC-0.5| >= this to pass univariate screen
NEAR_CONST_UNIQUE_RATIO = 1e-4
NEAR_CONST_VAR          = 1e-6
SPEARMAN_RHO_THRESH     = 0.95
MIN_OVERLAP_FOR_CORR    = 500

N_SLICES                 = 3          # rolling windows within TRAIN for stability
SHAP_RELATIVE_CUTOFF     = 0.05       # keep if mean|SHAP| >= 5% of top
TOP_K_FALLBACK           = 60
FAMILY_CAPS              = {"D": 2, "VEL": 2, "RATE": 3}  # caps per family

LOW_CARD = ["ProductCD", "card4", "card6", "DeviceType"]

PROTECT_ALWAYS: Set[str] = {
    "TransactionAmt", "hour_of_day", "day_of_week", "is_night", "is_weekend",
    "card1_amt_z_shrunk",
    "addr1_rate", "addr2_rate", "P_emaildomain_rate", "R_emaildomain_rate", "DeviceInfo_rate",
}

ID_TIME_TARGET = {"TransactionID", "TransactionDT", "isFraud"}


# ==========================
# Utilities / helpers
# ==========================
def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in df.columns:
        k = counts.get(c, 0)
        new_cols.append(c if k == 0 else f"{c}.{k}")
        counts[c] = k + 1
    df = df.copy()
    df.columns = new_cols
    return df

def is_flag(c: str) -> bool:
    return c.endswith("__is_missing") or c.startswith("is_")

def family_of(col: str) -> str:
    if col.startswith("D") and col[1:].split("_", 1)[0].isdigit():
        return "D"
    if col in {"mins_since_prev_card1","card1_txn_ct_1d","card1_txn_ct_7d","card1_amt_sum_1d","card1_amt_z_shrunk"}:
        return "VEL"
    if col.endswith("_rate"):
        return "RATE"
    return "OTHER"

def load_data(
    train_path: str = TRAIN_PATH,
    val_path: str = VAL_PATH,
    test_path: str = TEST_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = ensure_unique_columns(pd.read_parquet(train_path))
    val   = ensure_unique_columns(pd.read_parquet(val_path))
    test  = ensure_unique_columns(pd.read_parquet(test_path))
    sort_keys = ["TransactionDT", "TransactionID"]
    train = train.sort_values(sort_keys).reset_index(drop=True)
    val   = val.sort_values(sort_keys).reset_index(drop=True)
    test  = test.sort_values(sort_keys).reset_index(drop=True)
    return train, val, test

def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c, dt in df.dtypes.items():
        if c in ID_TIME_TARGET:
            continue
        if pd.api.types.is_numeric_dtype(dt):
            cols.append(c)
    return cols

def candidates_for_corr(train: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in numeric_columns(train):
        if is_flag(c) or c in PROTECT_ALWAYS or c in ID_TIME_TARGET:
            continue
        # treat obvious binaries as flags
        uniq = set(train[c].dropna().unique())
        if uniq.issubset({0, 1}):
            continue
        cols.append(c)
    return cols

def univariate_auc_screen(train: pd.DataFrame, auc_delta_min: float) -> Set[str]:
    keep: Set[str] = set()
    y = train["isFraud"].astype(int).values
    for c in numeric_columns(train):
        if c in ID_TIME_TARGET:
            continue
        x = train[c]
        if x.isna().all():
            continue
        x_f = x.fillna(x.median())
        if x_f.nunique() <= 1:
            continue
        try:
            auc = roc_auc_score(y, x_f)
        except Exception:
            continue
        auc = max(auc, 1 - auc)
        if (auc - 0.5) >= auc_delta_min:
            keep.add(c)

    keep |= PROTECT_ALWAYS
    keep |= {c for c in train.columns if is_flag(c)}
    keep |= set([c for c in LOW_CARD if c in train.columns])
    return keep

def near_constant_drop(train: pd.DataFrame, cols: List[str]) -> Set[str]:
    drop: Set[str] = set()
    n = len(train)
    for c in cols:
        s = train[c].dropna()
        if len(s) == 0:
            drop.add(c); continue
        if s.nunique() / max(n, 1) < NEAR_CONST_UNIQUE_RATIO:
            drop.add(c); continue
        if s.var() < NEAR_CONST_VAR:
            drop.add(c); continue
    return drop

def spearman_adj(train: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    def mask(col: str) -> pd.Series:
        f = f"{col}__is_missing"
        return (train[f] == 0) if f in train.columns else pd.Series(True, index=train.index)
    K = len(cols)
    adj = pd.DataFrame(False, index=cols, columns=cols)
    ranks = train[cols].rank()
    for i in range(K):
        ci = cols[i]; mi = mask(ci) & train[ci].notna()
        for j in range(i+1, K):
            cj = cols[j]; mj = mask(cj) & train[cj].notna()
            m = (mi & mj)
            if m.sum() < MIN_OVERLAP_FOR_CORR:
                continue
            r = ranks.loc[m, [ci, cj]].corr().iloc[0, 1]
            if abs(r) >= SPEARMAN_RHO_THRESH:
                adj.loc[ci, cj] = adj.loc[cj, ci] = True
    return adj

def mi_scores(train: pd.DataFrame, cols: List[str]) -> pd.Series:
    X = train[cols].copy().fillna(train[cols].median())
    y = train["isFraud"].astype(int).values
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return pd.Series(mi, index=cols)

def correlation_prune(train: pd.DataFrame, start_cols: Set[str]) -> Set[str]:
    cand = sorted([c for c in start_cols if (c not in PROTECT_ALWAYS) and (not is_flag(c)) and (c not in LOW_CARD)])
    drop_nc = near_constant_drop(train, cand)
    cand = [c for c in cand if c not in drop_nc]
    if not cand:
        return set(start_cols)

    adj = spearman_adj(train, cand)
    mi = mi_scores(train, cand)
    miss = train[cand].isna().mean()

    kept = set([c for c in start_cols if (c in PROTECT_ALWAYS) or is_flag(c) or (c in LOW_CARD)])
    visited: Set[str] = set()
    for c in cand:
        if c in visited:
            continue
        cluster = [c]; queue = [c]; visited.add(c)
        while queue:
            u = queue.pop()
            neighbors = list(adj.columns[adj.loc[u]])
            for v in neighbors:
                if v not in visited:
                    visited.add(v); queue.append(v); cluster.append(v)
        if len(cluster) == 1:
            kept.add(cluster[0]); continue
        rep = sorted(cluster, key=lambda x: (-mi.get(x, 0.0), miss.get(x, 1.0), len(x)))[0]
        kept.add(rep)
    return kept

# ------------- XGB / SHAP -------------
def build_matrix(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    cols = [c for c in feature_list if c not in ID_TIME_TARGET]
    low_present = [c for c in LOW_CARD if c in cols]

    num_cols = [c for c in cols if c not in low_present]
    X_num = df[num_cols].copy()

    for c in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[c]):
            X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    if low_present:
        X_cat = pd.get_dummies(df[low_present].astype("category"), drop_first=False, dtype=np.uint8)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    y = df["isFraud"].astype(int).values
    return X, y

def fit_xgb_and_rank(train_df: pd.DataFrame, val_df: pd.DataFrame, features_for_model: List[str]) -> pd.Series:
    Xtr, ytr = build_matrix(train_df, features_for_model)
    Xva, yva = build_matrix(val_df,   features_for_model)

    all_cols = sorted(set(Xtr.columns) | set(Xva.columns))
    Xtr = Xtr.reindex(columns=all_cols, fill_value=0)
    Xva = Xva.reindex(columns=all_cols, fill_value=0)

    feat_names = list(map(str, all_cols))
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_names)

    spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    params = dict(
        max_depth=4, eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        objective="binary:logistic", eval_metric="auc", tree_method="hist", scale_pos_weight=spw
    )
    booster = xgb.train(params, dtr, num_boost_round=1000,
                        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)

    shap_val = booster.predict(dva, pred_contribs=True)  # [n_val, n_features+1]; last is bias
    shap_abs_mean = np.abs(shap_val[:, :-1]).mean(axis=0)
    ranks = pd.Series(shap_abs_mean, index=feat_names).sort_values(ascending=False)
    return ranks

def rolling_stability_ranks(train: pd.DataFrame, features_for_model: List[str], n_slices: int) -> pd.Series:
    idx = train.sort_values("TransactionDT").index
    blocks = np.array_split(idx, n_slices*2)
    rank_list: List[pd.Series] = []
    for i in range(0, len(blocks), 2):
        tr_idx = np.concatenate(blocks[:i+1]) if i > 0 else blocks[0]
        va_idx = blocks[i+1] if (i+1) < len(blocks) else None
        if va_idx is None or len(va_idx) < 1000 or len(tr_idx) < 5000:
            continue
        ranks = fit_xgb_and_rank(train.loc[tr_idx], train.loc[va_idx], features_for_model)
        ranks_norm = ranks / (ranks.iloc[0] if ranks.iloc[0] > 0 else 1.0)
        rank_list.append(ranks_norm)
    if not rank_list:
        return pd.Series(dtype=float)
    R = pd.concat(rank_list, axis=1).fillna(0)
    R["avg"] = R.mean(axis=1)
    return R["avg"].sort_values(ascending=False)

# ------------- Selection -------------
def apply_family_caps(selected: List[str], family_caps: Dict[str, int]) -> List[str]:
    counts = {k: 0 for k in set(["D","VEL","RATE","OTHER"]) | set(family_caps.keys())}
    kept: List[str] = []
    for f in selected:
        fam = family_of(f)
        cap = family_caps.get(fam, None)
        if cap is None or counts[fam] < cap:
            kept.append(f)
            counts[fam] += 1
    return kept

def select_by_shap_threshold(shap_scores: pd.Series, rel_cutoff: float, topk_fallback: int) -> List[str]:
    if shap_scores.empty:
        return []
    top = shap_scores.iloc[0]
    if top <= 0:
        return []
    sel = shap_scores[shap_scores >= rel_cutoff * top].index.tolist()
    if len(sel) < 20 or len(sel) > 200:
        sel = shap_scores.head(topk_fallback).index.tolist()
    return sel


# ======================================================
# Public API: select_features (for Streamlit / imports)
# ======================================================
def select_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    auc_delta_min: float = AUC_DELTA_MIN,
    n_slices: int = N_SLICES,
    shap_relative_cutoff: float = SHAP_RELATIVE_CUTOFF,
    top_k_fallback: int = TOP_K_FALLBACK,
    family_caps: Optional[Dict[str, int]] = None,
    protect_always: Optional[Set[str]] = None,
    low_card: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Perform feature selection and return reduced (train, val, test) plus artifacts.
    """
    family_caps = family_caps or FAMILY_CAPS
    protect = set(protect_always or PROTECT_ALWAYS)
    low_card = list(low_card or LOW_CARD)

    # 1) Univariate AUC screen (train-only)
    pool = univariate_auc_screen(train_df, auc_delta_min)

    # 2) Redundancy prune (train-only)
    pool = correlation_prune(train_df, pool)

    # 3) SHAP ranking with stability + official VAL run
    features_for_model = sorted(list(pool - ID_TIME_TARGET))
    stab = rolling_stability_ranks(train_df, features_for_model, n_slices=n_slices)
    val_run = fit_xgb_and_rank(train_df, val_df, features_for_model)
    val_run_norm = val_run / (val_run.iloc[0] if len(val_run) and val_run.iloc[0] > 0 else 1.0)

    if stab.empty:
        combined = val_run_norm
    else:
        combined = pd.concat([stab.rename("stab"), val_run_norm.rename("val")], axis=1).fillna(0).mean(axis=1)
    combined = combined.sort_values(ascending=False)

    # 4) Threshold + family caps
    prelim = select_by_shap_threshold(combined, shap_relative_cutoff, top_k_fallback)
    prelim = apply_family_caps(prelim, family_caps)

    # 5) Final set to keep (preserve order as in train_df)
    final_keep = set(prelim) | protect | {c for c in train_df.columns if is_flag(c)} | (set(low_card) & set(train_df.columns))
    final_keep |= ID_TIME_TARGET
    final_keep_ordered = [c for c in train_df.columns if c in final_keep]

    # 6) Reduced frames
    train_red = train_df[final_keep_ordered].copy()
    val_red   = val_df[final_keep_ordered].copy()
    test_red  = test_df[final_keep_ordered].copy()

    artifacts = {
        "feature_keep": [c for c in final_keep_ordered if c not in ID_TIME_TARGET],
        "shap_scores": combined.sort_values(ascending=False),
        "prelim_selected": prelim,
        "family_caps": family_caps,
        "protect_always": sorted(list(protect)),
        "low_card": low_card,
        "counts": {
            "initial_train_cols": int(train_df.shape[1]),
            "kept_total_incl_ids": int(len(final_keep_ordered)),
            "kept_model_features": int(len(final_keep_ordered) - len(ID_TIME_TARGET)),
        }
    }

    if verbose:
        print(f"Initial columns: {train_df.shape[1]} | After selection: {len(final_keep_ordered)} (incl. id/time/target).")
        print(f"Kept (model features): {len(final_keep_ordered) - len(ID_TIME_TARGET)}")

    return train_red, val_red, test_red, artifacts


# ==========================
# File IO + CLI (zero-flag)
# ==========================
def _write_parquet(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def main_cli():
    parser = argparse.ArgumentParser(description="Feature selection for fraud dataset (AUC screen → redundancy prune → SHAP).")
    parser.add_argument("--in-train", type=str, default=TRAIN_PATH, help=f"Train FE parquet [default: {TRAIN_PATH}]")
    parser.add_argument("--in-val",   type=str, default=VAL_PATH,   help=f"Val   FE parquet [default: {VAL_PATH}]")
    parser.add_argument("--in-test",  type=str, default=TEST_PATH,  help=f"Test  FE parquet [default: {TEST_PATH}]")

    parser.add_argument("--out-train", type=str, default=OUT_TRAIN, help=f"Output reduced train [default: {OUT_TRAIN}]")
    parser.add_argument("--out-val",   type=str, default=OUT_VAL,   help=f"Output reduced val   [default: {OUT_VAL}]")
    parser.add_argument("--out-test",  type=str, default=OUT_TEST,  help=f"Output reduced test  [default: {OUT_TEST}]")
    parser.add_argument("--out-list",  type=str, default=OUT_LIST,  help=f"Output feature list JSON [default: {OUT_LIST}]")
    parser.add_argument("--out-ranks", type=str, default=OUT_RANKS, help=f"Output SHAP ranks CSV [default: {OUT_RANKS}]")
    parser.add_argument("--out-report",type=str, default=OUT_REPORT,help=f"Output family report CSV [default: {OUT_REPORT}]")

    parser.add_argument("--auc-delta-min", type=float, default=AUC_DELTA_MIN)
    parser.add_argument("--n-slices",       type=int,   default=N_SLICES)
    parser.add_argument("--shap-rel-cutoff",type=float, default=SHAP_RELATIVE_CUTOFF)
    parser.add_argument("--top-k-fallback", type=int,   default=TOP_K_FALLBACK)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    train, val, test = load_data(args.in_train, args.in_val, args.in_test)
    train_red, val_red, test_red, art = select_features(
        train, val, test,
        auc_delta_min=args.auc_delta_min,
        n_slices=args.n_slices,
        shap_relative_cutoff=args.shap_rel_cutoff,
        top_k_fallback=args.top_k_fallback,
        verbose=args.verbose,
    )

    _write_parquet(train_red, args.out_train)
    _write_parquet(val_red,   args.out_val)
    _write_parquet(test_red,  args.out_test)

    # Save artifacts
    Path(args.out_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_list, "w") as f:
        json.dump(art["feature_keep"], f)

    art["shap_scores"].to_csv(args.out_ranks, header=["mean_rel_shap"])
    pd.DataFrame({
        "feature": art["prelim_selected"],
        "family": [family_of(f) for f in art["prelim_selected"]],
    }).to_csv(args.out_report, index=False)

    if args.verbose:
        print(f"Wrote: {args.out_train}, {args.out_val}, {args.out_test}")
        print(f"Saved feature list to: {args.out_list}")
        print(f"Saved SHAP ranks to: {args.out_ranks}")
        print(f"Saved family report to: {args.out_report}")

if __name__ == "__main__":
    main_cli()
