import os, json
from typing import List, Set, Tuple, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

# ----------------------------
# Paths
# ----------------------------
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

# ----------------------------
# Config (tweakable)
# ----------------------------
# Univariate AUC threshold: keep if |AUC-0.5| >= this
AUC_DELTA_MIN = 0.01   # 1-pt better than random

# Redundancy thresholds
NEAR_CONST_UNIQUE_RATIO = 1e-4
NEAR_CONST_VAR          = 1e-6
SPEARMAN_RHO_THRESH     = 0.95
MIN_OVERLAP_FOR_CORR    = 500

# SHAP selection
N_SLICES                 = 3             # rolling windows within TRAIN for stability
SHAP_RELATIVE_CUTOFF     = 0.05          # keep if mean|SHAP| >= 5% of top
TOP_K_FALLBACK           = 60            # if threshold keeps too many/too few
FAMILY_CAPS              = {"D": 2, "VEL": 2, "RATE": 3}  # caps per family

# Low-card categoricals (one-hot for XGB)
LOW_CARD = ["ProductCD", "card4", "card6", "DeviceType"]

# Always keep for explainability
PROTECT_ALWAYS: Set[str] = {
    "TransactionAmt", "hour_of_day", "day_of_week", "is_night", "is_weekend",
    "card1_amt_z_shrunk",
    "addr1_rate", "addr2_rate", "P_emaildomain_rate", "R_emaildomain_rate", "DeviceInfo_rate",
}

ID_TIME_TARGET = {"TransactionID", "TransactionDT", "isFraud"}


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique (just in case) to avoid block-placement mismatches."""
    if df.columns.is_unique:
        return df
    counts = {}
    new_cols = []
    for c in df.columns:
        k = counts.get(c, 0)
        if k == 0:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}.{k}")
        counts[c] = k + 1
    df = df.copy()
    df.columns = new_cols
    return df


def is_flag(c: str) -> bool:
    return c.endswith("__is_missing") or c.startswith("is_")

def family_of(col: str) -> str:
    # For family caps
    if col.startswith("D") and col[1:].split("_",1)[0].isdigit():
        return "D"
    if col in {"mins_since_prev_card1","card1_txn_ct_1d","card1_txn_ct_7d","card1_amt_sum_1d","card1_amt_z_shrunk"}:
        return "VEL"
    if col.endswith("_rate"):
        return "RATE"
    return "OTHER"

# ----------------------------
# Utilities
# ----------------------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = ensure_unique_columns(pd.read_parquet(TRAIN_PATH))
    val   = ensure_unique_columns(pd.read_parquet(VAL_PATH))
    test  = ensure_unique_columns(pd.read_parquet(TEST_PATH))
    sort_keys = ["TransactionDT", "TransactionID"]
    train = train.sort_values(sort_keys).reset_index(drop=True)
    val   = val.sort_values(sort_keys).reset_index(drop=True)
    test  = test.sort_values(sort_keys).reset_index(drop=True)
    return train, val, test



def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c, dt in df.dtypes.items():
        if c in ID_TIME_TARGET: 
            continue
        if pd.api.types.is_numeric_dtype(dt):
            cols.append(c)
    return cols

def candidates_for_corr(df: pd.DataFrame) -> List[str]:
    # continuous numerics only, exclude flags, protected, ids/times/target and one-hot dummies
    cols = []
    for c in numeric_columns(df):
        if is_flag(c) or c in PROTECT_ALWAYS or c in ID_TIME_TARGET:
            continue
        # treat obvious binaries as flags (0/1)
        if set(df[c].dropna().unique()).issubset({0,1}):
            continue
        cols.append(c)
    return cols

def univariate_auc_screen(train: pd.DataFrame) -> Set[str]:
    keep = set()
    y = train["isFraud"].astype(int).values
    for c in numeric_columns(train):
        if c in ID_TIME_TARGET:
            continue
        x = train[c]
        if x.isna().all():
            continue
        # fill with train median for AUC calc (no labels used)
        x_f = x.fillna(x.median())
        # Skip constant
        if x_f.nunique() <= 1:
            continue
        try:
            auc = roc_auc_score(y, x_f)
        except Exception:
            # In rare cases with weird distributions; skip
            continue
        auc = max(auc, 1 - auc)  # distance from 0.5
        if (auc - 0.5) >= AUC_DELTA_MIN:
            keep.add(c)
    # Always add protected and flags
    keep |= PROTECT_ALWAYS
    keep |= {c for c in train.columns if is_flag(c)}
    # Keep low-card categoricals themselves (we'll one-hot them later for XGB)
    keep |= set([c for c in LOW_CARD if c in train.columns])
    return keep

def near_constant_drop(train: pd.DataFrame, cols: List[str]) -> Set[str]:
    drop = set()
    n = len(train)
    for c in cols:
        s = train[c].dropna()
        if len(s) == 0:
            drop.add(c); continue
        if s.nunique() / n < NEAR_CONST_UNIQUE_RATIO:
            drop.add(c); continue
        if s.var() < NEAR_CONST_VAR:
            drop.add(c); continue
    return drop

def spearman_adj(train: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Spearman via ranks; mask rows where feature is sentinel-missing, if a flag exists
    def mask(col):
        f = f"{col}__is_missing"
        return (train[f] == 0) if f in train.columns else pd.Series(True, index=train.index)
    K = len(cols)
    adj = pd.DataFrame(False, index=cols, columns=cols)
    ranks = train[cols].rank()  # rank handles ties
    for i in range(K):
        ci = cols[i]; mi = mask(ci) & train[ci].notna()
        for j in range(i+1, K):
            cj = cols[j]; mj = mask(cj) & train[cj].notna()
            m = (mi & mj)
            if m.sum() < MIN_OVERLAP_FOR_CORR: 
                continue
            r = ranks.loc[m, [ci, cj]].corr().iloc[0,1]
            if abs(r) >= SPEARMAN_RHO_THRESH:
                adj.loc[ci, cj] = adj.loc[cj, ci] = True
    return adj

def mi_scores(train: pd.DataFrame, cols: List[str]) -> pd.Series:
    X = train[cols].copy()
    # simple median fill for MI computation (no labels used)
    X = X.fillna(X.median())
    y = train["isFraud"].astype(int).values
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return pd.Series(mi, index=cols)

def correlation_prune(train: pd.DataFrame, start_cols: Set[str]) -> Set[str]:
    # From the start set, identify continuous candidates and prune clusters by MI + missingness
    cand = sorted([c for c in start_cols if (c not in PROTECT_ALWAYS) and (not is_flag(c)) and (c not in LOW_CARD)])
    # drop near-constant first
    drop_nc = near_constant_drop(train, cand)
    cand = [c for c in cand if c not in drop_nc]
    if not cand:
        return set(start_cols)

    adj = spearman_adj(train, cand)
    mi = mi_scores(train, cand)
    miss = train[cand].isna().mean()

    kept = set([c for c in start_cols if (c in PROTECT_ALWAYS) or is_flag(c) or (c in LOW_CARD)])
    visited = set()
    for c in cand:
        if c in visited: 
            continue
        # BFS cluster
        cluster = [c]; queue = [c]; visited.add(c)
        while queue:
            u = queue.pop()
            neighbors = list(adj.columns[adj.loc[u]])
            for v in neighbors:
                if v not in visited:
                    visited.add(v); queue.append(v); cluster.append(v)
        if len(cluster) == 1:
            kept.add(cluster[0]); continue
        # choose representative
        rep = sorted(
            cluster,
            key=lambda x: (-mi.get(x, 0.0), miss.get(x, 1.0), len(x))
        )[0]
        kept.add(rep)
    return kept

# ----------------------------
# XGB + SHAP ranking
# ----------------------------

def build_matrix(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return numeric design matrix X and label y. One-hot low-card cats, leave others numeric."""
    cols = [c for c in feature_list if c not in ID_TIME_TARGET]
    low_present = [c for c in LOW_CARD if c in cols]

    num_cols = [c for c in cols if c not in low_present]
    X_num = df[num_cols].copy()

    # Coerce any str/object numerics defensively
    for c in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[c]):
            X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    if low_present:
        # Stable one-hot (no drop_first); dtype small ints
        X_cat = pd.get_dummies(df[low_present].astype("category"),
                               drop_first=False, dtype=np.uint8)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num

    # Fill remaining NaNs with column medians (no labels used)
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    y = df["isFraud"].astype(int).values
    return X, y


def fit_xgb_and_rank(train_df: pd.DataFrame, val_df: pd.DataFrame, features_for_model: List[str]) -> pd.Series:
    # Build raw matrices
    Xtr, ytr = build_matrix(train_df, features_for_model)
    Xva, yva = build_matrix(val_df,   features_for_model)

    # 1) Align columns between TRAIN and VAL (union); fill missing with 0
    all_cols = sorted(set(Xtr.columns) | set(Xva.columns))
    Xtr = Xtr.reindex(columns=all_cols, fill_value=0)
    Xva = Xva.reindex(columns=all_cols, fill_value=0)

    # 2) XGBoost DMatrix — pass feature_names as list of strings
    feat_names = list(map(str, all_cols))
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_names)

    # 3) Train small model with early stop
    spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    params = dict(
        max_depth=4, eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        objective="binary:logistic", eval_metric="auc", tree_method="hist", scale_pos_weight=spw
    )
    booster = xgb.train(params, dtr, num_boost_round=1000,
                        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)

    # 4) SHAP via pred_contribs (last column is bias)
    shap_val = booster.predict(dva, pred_contribs=True)  # shape: [n_val, n_features+1]
    shap_abs_mean = np.abs(shap_val[:, :-1]).mean(axis=0)

    ranks = pd.Series(shap_abs_mean, index=feat_names).sort_values(ascending=False)
    return ranks


def rolling_stability_ranks(train: pd.DataFrame, features_for_model: List[str], n_slices: int) -> pd.Series:
    idx = train.sort_values("TransactionDT").index
    blocks = np.array_split(idx, n_slices*2)  # train/val mini windows
    rank_list = []
    for i in range(0, len(blocks), 2):
        tr_idx = np.concatenate(blocks[:i+1]) if i > 0 else blocks[0]
        va_idx = blocks[i+1] if (i+1) < len(blocks) else None
        if va_idx is None or len(va_idx) < 1000 or len(tr_idx) < 5000:
            continue
        ranks = fit_xgb_and_rank(train.loc[tr_idx], train.loc[va_idx], features_for_model)
        # normalize per run
        ranks_norm = ranks / (ranks.iloc[0] if ranks.iloc[0] > 0 else 1.0)
        rank_list.append(ranks_norm)
    if not rank_list:
        return pd.Series(dtype=float)
    R = pd.concat(rank_list, axis=1).fillna(0)
    R["avg"] = R.mean(axis=1)
    return R["avg"].sort_values(ascending=False)

# ----------------------------
# Selection logic
# ----------------------------
def apply_family_caps(selected: List[str]) -> List[str]:
    counts = {"D":0, "VEL":0, "RATE":0, "OTHER":0}
    kept = []
    for f in selected:
        fam = family_of(f)
        cap = FAMILY_CAPS.get(fam, None)
        if cap is None or counts[fam] < cap:
            kept.append(f)
            counts[fam] += 1
    return kept

def select_by_shap_threshold(shap_scores: pd.Series) -> List[str]:
    if shap_scores.empty:
        return []
    top = shap_scores.iloc[0]
    if top <= 0:
        return []
    sel = shap_scores[shap_scores >= SHAP_RELATIVE_CUTOFF * top].index.tolist()
    # sanity fallback
    if len(sel) < 20 or len(sel) > 200:
        sel = shap_scores.head(TOP_K_FALLBACK).index.tolist()
    return sel

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    train, val, test = load_data()

    # 1) Univariate AUC screen (train-only) -> start pool
    pool = univariate_auc_screen(train)

    # 2) Redundancy prune (train-only)
    pool = correlation_prune(train, pool)

    # 3) SHAP ranking with stability (train rolling windows) + official VAL run
    # Build features_for_model = pool + LOW_CARD one-hots handled inside build_matrix
    features_for_model = sorted(list(pool - ID_TIME_TARGET))
    stab = rolling_stability_ranks(train, features_for_model, n_slices=N_SLICES)

    # Also add a run where we train on full TRAIN, early-stop on VAL, SHAP on VAL
    val_run = fit_xgb_and_rank(train, val, features_for_model)
    val_run_norm = val_run / (val_run.iloc[0] if val_run.iloc[0] > 0 else 1.0)

    # Combine (outer join, mean of available)
    if stab.empty:
        combined = val_run_norm
    else:
        combined = pd.concat([stab.rename("stab"), val_run_norm.rename("val")], axis=1).fillna(0).mean(axis=1)
    combined = combined.sort_values(ascending=False)

    # 4) Threshold selection by relative SHAP + family caps
    prelim = select_by_shap_threshold(combined)
    prelim = apply_family_caps(prelim)

    # 5) Build final feature list to KEEP for modeling / rule mining
    # Always include IDs/time/target when saving parquet; keep all flags and low-card categoricals
    final_keep = set(prelim) | PROTECT_ALWAYS | {c for c in train.columns if is_flag(c)} | (set(LOW_CARD) & set(train.columns))
    final_keep |= ID_TIME_TARGET  # mandatory
    final_keep = [c for c in train.columns if c in final_keep]  # preserve original order

    # 6) Save artifacts
    json.dump([c for c in final_keep if c not in ID_TIME_TARGET], open(OUT_LIST, "w"))
    combined.to_csv(OUT_RANKS, header=["mean_rel_shap"])

    # Optional: export a simple correlation cluster report (already pruned) — just list families & notes
    pd.DataFrame({
        "feature": prelim,
        "family": [family_of(f) for f in prelim]
    }).to_csv(OUT_REPORT, index=False)

    # 7) Write reduced Parquets
    train[final_keep].to_parquet(OUT_TRAIN)
    val[final_keep].to_parquet(OUT_VAL)
    test[final_keep].to_parquet(OUT_TEST)


    print(f"Initial columns: {train.shape[1]} | After selection: {len(final_keep)} (incl. id/time/target).")
    print(f"Kept (model features): {len(final_keep) - len(ID_TIME_TARGET)}")
    print(f"Wrote: {OUT_TRAIN}, {OUT_VAL}, {OUT_TEST}")
    print(f"Saved feature list to: {OUT_LIST}")
    print(f"Saved SHAP ranks to: {OUT_RANKS}")

if __name__ == "__main__":
    main()
