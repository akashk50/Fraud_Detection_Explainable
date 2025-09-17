from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os
import helperFuncs

# Get the directory this script is in
CURRENT_DIR = os.path.dirname(__file__)

# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Build the path to the data file
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ieee_fraud_features_encoded.parquet")


AMOUNT_COL = "TransactionAmt"
TIME_COL = "TransactionDT"
TARGET = "isFraud"

def run_miner(
    df: pd.DataFrame,
    *,
    n_bootstraps: int = 100,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    bootstrap_ratio: int = 3,
    min_votes_post: int = 5,
    snap_step: int = 10
    ): 
    max_depth = 3
    min_sample_leaf = 5

    GROUP_MAP = {
        "ProductCD_": "ProductCD",
        "card6_": "card6",
        "card4_": "card4",
    }

    # (Optional) full level lists per group if you want NOT IN for negative-only rules.
    # Example enumerations; adjust to match your encoding.
    GROUP_LEVELS = {
        "ProductCD_": ["W","R","H","S","C"],
        "card6_": ["credit","debit","charge","debit_or_credit"],
        "card4_": ["visa","mastercard","discover","american_express"],
    }

    df = df.drop(columns=["card1", "card2", "card3", "card5"], errors="ignore", axis=1)

    n_bootstraps = 100

    split = helperFuncs.sampleSplit(df=df, TIME_COL=TIME_COL, TARGET=TARGET)

    X_train       = split["X_train"]
    y_train       = split["y_train"]
    X_val         = split["X_val"]
    y_val         = split["y_val"]
    fraud_idx     = split["fraud_idx"]    # numpy array of int indices
    ok_idx        = split["ok_idx"]       # numpy array of int indices
    feature_names = split["feature_names"]



    # 2) Bootstrap loop → fit shallow tree → extract raw leaf paths
    rules_tally = {}  # dict keyed by canonical rule; value holds {"votes": int, "clauses": ...}

    for i in range(n_bootstraps):
        # 2a) draw bootstrap indices from the *train* set
        idx = helperFuncs.stratified_bootstrap_ratio(fraud_idx=fraud_idx, ok_idx=ok_idx, ratio=1)

        # 2b) subset
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]

        # 2c) fit depth-2 tree
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_sample_leaf, random_state=42 + i)
        tree.fit(X_boot, y_boot)

        # 2d) extract raw paths (use the dfs you wrote)
        leaves = helperFuncs.dfs_leaf_paths(tree, feature_names, positive_class=1, keep_all=False)

        # 2e) for each leaf path: clean → canonicalize → tally
        for leaf in leaves:
            raw_path = leaf["path"]

            # CLEAN & MERGE (your functions; placeholders here)
            norm_clauses = helperFuncs.clean_and_merge(raw_path)       # snap $10, integerize hour/day, one-hot @0.5
            if not norm_clauses:
                continue

            key = helperFuncs.canonical_key(norm_clauses)              # sorted, hashable tuple

            rec = rules_tally.get(key)
            if rec is None:
                rules_tally[key] = {"votes": 1, "clauses": norm_clauses}
            else:
                rec["votes"] += 1

    # 3) Turn into a list sorted by votes when you’re ready to export
    candidates = sorted(
        ({"key": k, "votes": v["votes"], "clauses": v["clauses"]} for k, v in rules_tally.items()),
        key=lambda d: d["votes"],
        reverse=True
    )

    cleaned = helperFuncs.postprocess_rules(candidates, min_votes=5, snap_step=10)

    rule_eval = helperFuncs.evaluate_rules(rules = cleaned, X_val=X_val, y_val=y_val, min_support=20)

    baseline_val = df["isFraud"].mean()  # fraction of rows where isFraud == 1
    sql_outputs = []
    english_outputs = []

    for _, row in rule_eval.iterrows():
        clauses = row["rule"]
        sql_where, english = helperFuncs.clauses_to_sql_where_pretty(clauses)
        sql_outputs.append(sql_where)
        english_outputs.append(english)

    # Add back into your DataFrame
    rule_eval["sql_where"] = sql_outputs
    rule_eval["english"] = english_outputs

    return rule_eval, cleaned, baseline_val

# --- keep script mode for convenience ---
if __name__ == "__main__":
    # Running directly? Fall back to your project data path.
    CURRENT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ieee_fraud_features_encoded.parquet")

    df = pd.read_parquet(DATA_PATH)
    # (optionally drop columns like before if you want)
    # df = df.drop(columns=["card1","card2","card3","card5"], errors="ignore")

    rule_eval, cleaned, baseline_val = run_miner(df)
    print(rule_eval.head())
    print("Baseline (val):", baseline_val)