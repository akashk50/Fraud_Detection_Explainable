# scripts/decisionTree.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import helperFuncs  # relies on app adding ./scripts to sys.path

TARGET = "isFraud"

# --------------------------------
# Project-relative default paths
# --------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "fraud_train_reduced.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "fraud_val_reduced.parquet")

# --------------------------------
# Core miner
# --------------------------------
def run_miner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_bootstraps: int = 1000,
    max_depth: int = 5,
    min_samples_leaf: int = 5,
    min_votes_post: int = 5,
    bootstrap_ratio: int = 1,     # 1:1 nonfraud:fraud in each bootstrap
    seed: int = 42
) -> Tuple[pd.DataFrame, list[Dict[str, Any]], float]:
    """
    Bootstrap-mined decision-tree leaves → rules → postprocess → evaluate on X_val/y_val.
    Returns: (rule_eval_df, cleaned_rules, baseline_val_rate)
    """
    feature_names = X_train.columns.tolist()

    # Build index pools for stratified bootstrap
    fraud_idx = y_train[y_train == 1].index.to_numpy()
    ok_idx    = y_train[y_train == 0].index.to_numpy()

    rules_tally: Dict[str, Dict[str, Any]] = {}
    rng = np.random.default_rng(seed)

    for i in range(n_bootstraps):
        # draw bootstrapped indices (fraud with replacement, ok with replacement at `ratio`)
        K = len(fraud_idx)
        if K == 0 or len(ok_idx) == 0:
            break  # nothing to sample
        M = max(bootstrap_ratio * K, 1)
        fraud_sample = rng.choice(fraud_idx, size=K, replace=True)
        ok_sample    = rng.choice(ok_idx,    size=M, replace=True)
        idx = rng.permutation(np.concatenate([fraud_sample, ok_sample]))

        X_boot = X_train.loc[idx]
        y_boot = y_train.loc[idx]

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=1234 + i
        )
        tree.fit(X_boot, y_boot)

        leaves = helperFuncs.dfs_leaf_paths(tree, feature_names, positive_class=1, keep_all=False)

        for leaf in leaves:
            norm = helperFuncs.clean_and_merge(leaf["path"])
            if not norm:
                continue
            key = helperFuncs.canonical_key(norm)
            rec = rules_tally.get(key)
            if rec is None:
                rules_tally[key] = {"votes": 1, "clauses": norm}
            else:
                rec["votes"] += 1

    candidates = [
        {"key": k, "votes": v["votes"], "clauses": v["clauses"]}
        for k, v in rules_tally.items()
    ]
    cleaned = helperFuncs.postprocess_rules(candidates, min_votes=min_votes_post)

    rule_eval = helperFuncs.evaluate_rules(cleaned, X_val, y_val, min_support=20)

    # render SQL + English
    sql_where_list, english_list = [], []
    for _, row in rule_eval.iterrows():
        sql_where, english = helperFuncs.clauses_to_sql_where_pretty(row["rule"])
        sql_where_list.append(sql_where)
        english_list.append(english)
    rule_eval["sql_where"] = sql_where_list
    rule_eval["english"] = english_list

    baseline_val = float(y_val.mean())
    return rule_eval, cleaned, baseline_val

# --------------------------------
# Convenience wrappers for Streamlit / legacy calls
# --------------------------------
def _xy_from_df(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target].astype(int)
    X = df.drop(columns=[target], errors="ignore")
    return X, y

def run_miner_from_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    **kwargs
) -> Tuple[pd.DataFrame, list[Dict[str, Any]], float]:
    """
    Convenience wrapper: accept full train/val frames (with target col) and mine rules.
    """
    X_train, y_train = _xy_from_df(train_df)
    X_val, y_val     = _xy_from_df(val_df)
    return run_miner(X_train, y_train, X_val, y_val, **kwargs)

def run_miner_val_only(
    val_df: pd.DataFrame,
    **kwargs
) -> Tuple[pd.DataFrame, list[Dict[str, Any]], float]:
    """
    Legacy-style helper: mine using VAL both to bootstrap and to evaluate.
    Handy if your app currently only passes a single labeled dataframe.
    """
    X_val, y_val = _xy_from_df(val_df)
    return run_miner(X_val, y_val, X_val, y_val, **kwargs)

# --------------------------------
# CLI (zero-flag friendly)
# --------------------------------
def _read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def main_cli():
    parser = argparse.ArgumentParser(description="Mine fraud rules via bootstrapped decision trees.")
    parser.add_argument("--train", type=str, default=TRAIN_PATH, help=f"Train reduced parquet [default: {TRAIN_PATH}]")
    parser.add_argument("--val",   type=str, default=VAL_PATH,   help=f"Val   reduced parquet [default: {VAL_PATH}]")
    parser.add_argument("--n-bootstraps", type=int, default=100, help="Number of bootstraps")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-votes-post", type=int, default=5)
    parser.add_argument("--bootstrap-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-top", type=int, default=20, help="Rows to print from rule catalog")
    args = parser.parse_args()

    df_tr = _read_parquet(args.train)
    df_va = _read_parquet(args.val)

    rule_eval, cleaned, baseline_val = run_miner_from_frames(
        df_tr, df_va,
        n_bootstraps=args.n_bootstraps,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_votes_post=args.min_votes_post,
        bootstrap_ratio=args.bootstrap_ratio,
        seed=args.seed,
    )

    print(rule_eval.head(args.print_top).to_string(index=False))
    print("\nBaseline (VAL fraud rate):", f"{baseline_val:.4%}")

if __name__ == "__main__":
    main_cli()
