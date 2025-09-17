import numpy as np
from statsmodels.stats.proportion import proportion_confint
import pandas as pd
import math



# Split fraud and nonfraud in a time series safe way

def sampleSplit(df, TIME_COL, TARGET):
    # 1) time-safe sort
    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)

    # 2) 80/20 split by time
    cut = int(len(df_sorted) * 0.8)
    train = df_sorted.iloc[:cut].copy()
    val   = df_sorted.iloc[cut:].copy()

    # 3) Build X/y — drop TARGET and TIME_COL from X_* (keep hour/day features you derived)
    X_train = train.drop(columns=[TARGET, TIME_COL], errors="ignore")
    y_train = train[TARGET].astype(int)

    X_val = val.drop(columns=[TARGET, TIME_COL], errors="ignore")
    y_val = val[TARGET].astype(int)

    feature_names = X_train.columns.to_list()

    # Stratified bootstrap indices from TRAIN ONLY
    fraud_idx = y_train[y_train == 1].index.to_numpy()
    ok_idx    = y_train[y_train == 0].index.to_numpy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "fraud_idx": fraud_idx,
        "ok_idx": ok_idx,
        "feature_names": feature_names,
    }


# Amount snapping: round thresholds to nearest $100
def snap_amount_threshold(x, step=100):
    if np.isfinite(x):
        return float(int(round(x / step) * step))
    return x

# Treat one-hot splits as == 1 if threshold ~ 0.5
def is_one_hot_split(feature_name, threshold):
    # heuristic: one-hot features are 0/1 and split near 0.5
    return threshold > 0.0 and threshold < 1.0


def stratified_bootstrap_ratio(fraud_idx, ok_idx, ratio=3, rng=None):   #  Builds bootstrapped sample of fraud:nonfraud
    if rng is None:
        rng = np.random.default_rng()

    K = len(fraud_idx)
    M = max(ratio * K, 1)  # ensure at least one non-fraud

    fraud_sample = rng.choice(fraud_idx, size=K, replace=True)
    ok_sample    = rng.choice(ok_idx,    size=M, replace=True)

    idx = np.concatenate([fraud_sample, ok_sample])
    rng.shuffle(idx)
    return idx


from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def dfs_leaf_paths(
    estimator,
    feature_names: List[str],
    positive_class: int = 1,
    keep_all: bool = False
) -> List[Dict[str, Any]]:
    """
    Traverse a fitted sklearn DecisionTreeClassifier and collect raw root→leaf paths.

    Parameters
    ----------
    estimator : sklearn.tree.DecisionTreeClassifier
        A fitted decision tree classifier (depth=2 in your use case).
    feature_names : List[str]
        Names aligned to estimator.tree_.feature indices.
    positive_class : int, default=1
        Class label considered "positive" (fraud). Used when keep_all=False.
    keep_all : bool, default=False
        If True, return all leaves. If False, return only leaves whose predicted class == positive_class.

    Returns
    -------
    List[Dict[str, Any]]
        Each item corresponds to one leaf:
        {
          "leaf_id": int,
          "path": [ {"feature": str, "op": "<=" or ">", "threshold": float, "feature_idx": int}, ... ],
          "n_node_samples": int,                  # in-bag samples at the leaf
          "class_counts": List[int],              # counts per class
          "proba": List[float],                   # normalized class probabilities
          "predicted_class": int
        }
    """
    tree = estimator.tree_
    children_left  = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value  # shape (n_nodes, 1, n_classes)

    n_nodes = tree.node_count
    n_classes = values.shape[-1]

    # Pre-map feature index → name (sklearn uses -2 for leaves)
    feat_name = {i: feature_names[i] for i in range(len(feature_names))}

    def is_leaf(node_id: int) -> bool:
        return children_left[node_id] == children_right[node_id]  # both -1

    def leaf_meta(node_id: int) -> Dict[str, Any]:
        counts = values[node_id][0]
        total = counts.sum()
        proba = (counts / total) if total > 0 else np.zeros_like(counts, dtype=float)
        pred_class = int(np.argmax(counts))
        return {
            "n_node_samples": int(tree.n_node_samples[node_id]),
            "class_counts": counts.astype(int).tolist(),
            "proba": proba.tolist(),
            "predicted_class": pred_class,
        }

    paths: List[Dict[str, Any]] = []

    # Stack holds tuples: (node_id, path_so_far)
    # path_so_far is a list of raw clause dicts
    stack: List[tuple[int, List[Dict[str, Any]]]] = [(0, [])]  # start at root (node 0)

    while stack:
        node_id, path_so_far = stack.pop()

        if is_leaf(node_id):
            meta = leaf_meta(node_id)
            if keep_all or meta["predicted_class"] == positive_class:
                paths.append({
                    "leaf_id": node_id,
                    "path": path_so_far,  # raw clauses in traversal order
                    **meta,
                })
            continue

        # Internal node: create left/right clauses and push children
        f_idx = features[node_id]
        thr = thresholds[node_id]

        # Defensive: skip if feature index is invalid (shouldn't happen in a fitted tree)
        if f_idx < 0 or f_idx >= len(feature_names):
            # Still traverse children without adding a clause (rare/degenerate)
            if children_right[node_id] != -1:
                stack.append((children_right[node_id], list(path_so_far)))
            if children_left[node_id] != -1:
                stack.append((children_left[node_id], list(path_so_far)))
            continue

        f_name = feat_name[f_idx]

        # Left branch: feature <= threshold
        left_clause = {
            "feature": f_name,
            "op": "<=",
            "threshold": float(thr),
            "feature_idx": int(f_idx),
        }
        # Right branch: feature > threshold
        right_clause = {
            "feature": f_name,
            "op": ">",
            "threshold": float(thr),
            "feature_idx": int(f_idx),
        }

        # Push right then left so that left is processed first (optional ordering)
        if children_right[node_id] != -1:
            stack.append((children_right[node_id], path_so_far + [right_clause]))
        if children_left[node_id] != -1:
            stack.append((children_left[node_id], path_so_far + [left_clause]))

    return paths


INTEGER_LIKE_FEATURES = {"hour_of_day", "day_of_week"}
AMOUNT_FEATURES = {"TransactionAmt"}  # extend if needed

def clean_and_merge(
    raw_path: List[Dict[str, Any]],
    integer_like: set = INTEGER_LIKE_FEATURES,
    amount_features: set = AMOUNT_FEATURES,
    one_hot_threshold_low: float = 0.0,
    one_hot_threshold_high: float = 1.0,
) -> List[Dict[str, Any]] | None:
    """
    Normalize and merge a raw root→leaf path (list of {'feature','op','threshold',...}).
    - One-hot splits near 0.5 → equality on 0/1
    - Integer-like (hour/day) → inclusive integer ranges
    - Amounts → snap thresholds to nearest $10
    - Continuous others → keep raw thresholds (no fancy snapping)
    - Merge repeated feature constraints
    Returns a list of normalized clauses or None if contradictory/empty.
    """
    # Accumulators per feature
    eq_constraints: Dict[str, int] = {}  # for one-hots: feature -> 0 or 1
    int_ranges: Dict[str, Dict[str, int]] = {}  # feature -> {'low': int|None, 'high': int|None}
    cont_ranges: Dict[str, Dict[str, Any]] = {}  # feature -> {'low': float|None, 'low_inc': bool, 'high': float|None, 'high_inc': bool}

    def add_one_hot_eq(feat: str, val: int) -> bool:
        prev = eq_constraints.get(feat)
        if prev is not None and prev != val:
            return False  # contradiction
        eq_constraints[feat] = val
        return True

    def add_int_bound(feat: str, op: str, thr: float) -> bool:
        # map to inclusive integer bounds
        if op == "<=":
            hi = int(np.floor(thr))
            rng = int_ranges.setdefault(feat, {"low": None, "high": None})
            rng["high"] = hi if rng["high"] is None else min(rng["high"], hi)
        else:  # ">"
            lo = int(np.ceil(thr))
            rng = int_ranges.setdefault(feat, {"low": None, "high": None})
            rng["low"] = lo if rng["low"] is None else max(rng["low"], lo)
        return True

    def add_cont_bound(feat: str, op: str, thr: float, snap_amount: bool) -> bool:
        # Optionally snap amounts to $10
        if snap_amount:
            thr = snap_amount_threshold(thr, step=10)
        rng = cont_ranges.setdefault(feat, {"low": None, "low_inc": False, "high": None, "high_inc": False})
        if op == "<=":
            if rng["high"] is None or thr < rng["high"]:
                rng["high"] = float(thr)
                rng["high_inc"] = True
            elif thr == rng["high"]:
                rng["high_inc"] = rng["high_inc"] or True
        else:  # ">"
            if rng["low"] is None or thr > rng["low"]:
                rng["low"] = float(thr)
                rng["low_inc"] = False  # '>' implies open lower bound
            elif thr == rng["low"]:
                rng["low_inc"] = rng["low_inc"] and False
        return True

    # Pass 1: interpret each raw clause
    for clause in raw_path:
        feat = clause["feature"]
        op = clause["op"]          # "<=" or ">"
        thr = float(clause["threshold"])

        # One-hot heuristic: thresholds strictly between 0 and 1 → 0/1 equality
        if is_one_hot_split(feat, thr):
            val = 0 if op == "<=" else 1
            if not add_one_hot_eq(feat, val):
                return None  # contradiction
            continue

        # Integer-like features
        if feat in integer_like:
            if not add_int_bound(feat, op, thr):
                return None
            continue

        # Continuous features
        snap_amt = feat in amount_features
        if not add_cont_bound(feat, op, thr, snap_amount=snap_amt):
            return None

    # Pass 2: validate ranges
    # Integer-like: ensure low <= high if both present
    for feat, rng in int_ranges.items():
        lo, hi = rng["low"], rng["high"]
        if lo is not None and hi is not None and lo > hi:
            return None  # empty set after merging

    # Continuous: ensure consistent low/high
    for feat, rng in cont_ranges.items():
        lo, hi = rng["low"], rng["high"]
        if lo is not None and hi is not None:
            if lo > hi:
                return None
            if lo == hi and (not rng["low_inc"] or not rng["high_inc"]):
                # open interval at a point → empty
                return None

    # Pass 3: build normalized clauses list
    norm: List[Dict[str, Any]] = []

    # One-hots to eq clauses
    for feat, val in eq_constraints.items():
        norm.append({"feature": feat, "type": "eq", "value": int(val)})

    # Integer ranges
    for feat, rng in int_ranges.items():
        lo = rng["low"] if rng["low"] is not None else -np.inf
        hi = rng["high"] if rng["high"] is not None else np.inf
        # If both are infinite, this constraint is redundant; skip
        if np.isinf(lo) and np.isinf(hi):
            continue
        # Clip to valid domain if you want (e.g., 0–23 for hour); MVP: keep as is
        low_int = None if np.isinf(lo) else int(lo)
        high_int = None if np.isinf(hi) else int(hi)
        norm.append({"feature": feat, "type": "int_range", "low": low_int, "high": high_int})

    # Continuous ranges
    for feat, rng in cont_ranges.items():
        lo = rng["low"]
        hi = rng["high"]
        low_inc = rng["low_inc"]
        high_inc = rng["high_inc"]
        # If both None, skip; else emit
        if lo is None and hi is None:
            continue
        norm.append({
            "feature": feat,
            "type": "range",
            "low": lo,
            "low_inclusive": bool(low_inc) if lo is not None else False,
            "high": hi,
            "high_inclusive": bool(high_inc) if hi is not None else False,
        })

    # If nothing meaningful left, return None
    if not norm:
        return None

    # Sort deterministically for nicer diffs/readability
    norm.sort(key=lambda d: (d["feature"], d["type"]))
    return norm


def canonical_key(norm_clauses: List[Dict[str, Any]]) -> Tuple:
    """
    Produce a stable, hashable key from normalized clauses for deduplication.
    """
    key_parts = []
    for c in norm_clauses:
        t = c["type"]
        f = c["feature"]
        if t == "eq":
            key_parts.append((f, "eq", int(c["value"])))
        elif t == "int_range":
            # None markers are important to keep identity (open bounds)
            key_parts.append((f, "int_range", c.get("low", None), c.get("high", None)))
        elif t == "range":
            key_parts.append((
                f, "range",
                c.get("low", None), c.get("low_inclusive", False),
                c.get("high", None), c.get("high_inclusive", False),
            ))
        else:
            # Fallback: include raw dict items in deterministic order
            key_parts.append((f, t, tuple(sorted((k, v) for k, v in c.items() if k not in {"feature", "type"}))))
    # Sort key parts to make the overall key independent of clause order
    key_parts.sort()
    return tuple(key_parts)


from collections import defaultdict
from math import isfinite

def postprocess_rules(
    candidates,
    *,
    min_votes=5,
    amt_feature="TransactionAmt",
    snap_step=10,
    amt_tol=20,  # if you later want to collapse bands (not used in weighted average path)
    behavioral_features=("TransactionAmt", "hour_of_day", "day_of_week"),
    onehot_prefixes=("ProductCD_", "card6_", "card4_"),
    prefer_direction="gt",  # if both <=T and >T exist for same base, keep which side ("gt" or "le")
    require_behavior=True,  # drop rules that have no behavioral feature
):
    """
    Post-process mined rules:
    1) drop low-vote rules and product-only tautologies;
    2) compress one-hot groups (keep a single positive equality per group, drop negatives);
    3) group rules by their non-amount 'base' and direction (gt / le),
       then merge thresholds via weighted average (weights = votes);
    4) snap the canonical threshold and emit a cleaned rule per group.

    Returns: list of cleaned candidate dicts (same shape as input: {"votes", "clauses"})
    """

    # --- helpers --------------------------------------------------------------

    def is_onehot(fname):
        return any(fname.startswith(p) for p in onehot_prefixes)

    def onehot_group_name(fname):
        for p in onehot_prefixes:
            if fname.startswith(p):
                return p  # group by prefix
        return None

    def has_behavioral(clauses):
        return any(c.get("feature") in behavioral_features for c in clauses)

    def normalize_amount_clause(clause):
        """
        Map a 'range' clause into (direction, threshold):
          - low only => ('gt', low)
          - high only => ('le', high)
        Return None if not a simple one-sided range.
        """
        if clause.get("type") != "range" or clause.get("feature") != amt_feature:
            return None
        low = clause.get("low")
        high = clause.get("high")
        if low is not None and high is None and isfinite(low):
            return ("gt", float(low))
        if high is not None and low is None and isfinite(high):
            return ("le", float(high))
        # If it's a proper band (both bounds), treat separately (you can extend if you want bands)
        return None

    def snap_amount(x):
        if x is None:
            return None
        return float(round(x / snap_step) * snap_step)

    def compress_onehots(clauses):
        """
        Keep at most ONE positive equality per one-hot group.
        Drop all '== 0' negatives if the group has any positive ==1.
        Keep non one-hot clauses as-is.
        """
        kept = []
        chosen_positive = {}  # group -> feature kept

        # First pass: collect any positive ==1 per group
        for c in clauses:
            f = c.get("feature")
            if is_onehot(f) and c.get("type") == "eq" and c.get("value") == 1:
                g = onehot_group_name(f)
                if g not in chosen_positive:
                    chosen_positive[g] = f

        # Second pass: emit
        for c in clauses:
            f = c.get("feature")
            if is_onehot(f):
                g = onehot_group_name(f)
                if f == chosen_positive.get(g):
                    kept.append(c)  # keep the single positive equality
                else:
                    # drop negatives and extra one-hots from same group
                    continue
            else:
                kept.append(c)
        return kept

    def drop_tautology(clauses):
        """
        Drop rules that are ONLY one-hot eq==0 style (after compression,
        that means: no clauses left or no behavioral + no positive equality).
        After compression we only retain positives and non-onehot, so
        a 'product-only tautology' will typically end up as [] or only non-behavioral.
        We also optionally require a behavioral clause overall.
        """
        if require_behavior and not has_behavioral(clauses):
            return True
        return False

    def base_key_without_amount(clauses):
        """
        Build a hashable key for all non-amount clauses.
        We sort by (feature, type, value) for stability.
        """
        parts = []
        for c in clauses:
            if c.get("feature") == amt_feature:
                continue
            if c.get("type") == "eq":
                parts.append((c["feature"], "eq", c["value"]))
            elif c.get("type") == "range":
                # keep non-amount ranges too if you ever add hour/day ranges
                parts.append((c["feature"], "range",
                              c.get("low"), c.get("low_inclusive", False),
                              c.get("high"), c.get("high_inclusive", False)))
            else:
                parts.append((c["feature"], c.get("type"), None))
        return tuple(sorted(parts))

    # --- accumulate per (base_key, direction) --------------------------------

    # We’ll merge thresholds by weighted average (weights = votes)
    agg = defaultdict(lambda: {
        "w_sum": 0.0,    # sum(weights * threshold)
        "w": 0.0,        # sum(weights)
        "votes": 0,      # total votes across merged rules
        "any_clauses": None,  # to reconstruct the base clauses later
        "dir": None,     # "gt" or "le"
    })

    # Optionally store both sides to later decide to keep only one direction
    by_base = defaultdict(lambda: {"gt": None, "le": None})

    for cand in candidates:
        votes = int(cand.get("votes", 0))
        if votes < min_votes:
            continue

        clauses = cand.get("clauses", [])
        # 1) compress one-hot groups
        clauses = compress_onehots(clauses)

        # 2) remove product-only tautologies / non-behavioral if required
        if drop_tautology(clauses):
            continue

        # 3) find a simple one-sided amount clause (skip if none)
        amt_clause = None
        for c in clauses:
            if c.get("feature") == amt_feature and c.get("type") == "range":
                amt_clause = c
                break

        dir_thr = normalize_amount_clause(amt_clause) if amt_clause else None
        if dir_thr is None:
            # if no usable amount clause, skip (keeps catalog behavioral)
            continue

        direction, thr = dir_thr

        # 4) build base key (non-amount clauses)
        base_key = base_key_without_amount(clauses)

        # 5) accumulate weighted average per (base_key, direction)
        key = (base_key, direction)
        rec = agg[key]
        rec["w_sum"] += votes * thr
        rec["w"]     += votes * 1.0
        rec["votes"] += votes
        rec["dir"]    = direction
        if rec["any_clauses"] is None:
            # store non-amount clauses to rebuild later
            base_clauses = [c for c in clauses if c.get("feature") != amt_feature]
            rec["any_clauses"] = base_clauses

        # mark presence by base (for later direction selection)
        if by_base[base_key][direction] is None:
            by_base[base_key][direction] = True

    # --- build merged rules ---------------------------------------------------

    merged = []
    for (base_key, direction), rec in agg.items():
        if rec["w"] <= 0:
            continue
        thr_avg = rec["w_sum"] / rec["w"]
        thr_avg = snap_amount(thr_avg)

        # If both directions exist for this base, optionally keep only one side
        both_exist = (by_base[base_key]["gt"] and by_base[base_key]["le"])
        if both_exist and direction != prefer_direction:
            # skip the non-preferred side (simple MVP heuristic)
            continue

        # Rebuild clauses
        new_clauses = list(rec["any_clauses"])  # copy
        if direction == "gt":
            amt_clause = {"feature": amt_feature, "type": "range",
                          "low": float(thr_avg), "low_inclusive": False,
                          "high": None, "high_inclusive": False}
        else:  # "le"
            amt_clause = {"feature": amt_feature, "type": "range",
                          "low": None, "low_inclusive": False,
                          "high": float(thr_avg), "high_inclusive": True}
        new_clauses.append(amt_clause)

        merged.append({
            "votes": rec["votes"],
            "clauses": new_clauses
        })

    # Sort by votes desc, then by threshold (descending for "gt", ascending for "le")
    def sort_key(c):
        # find amount to stabilize sorting
        thr = None
        for cl in c["clauses"]:
            if cl.get("feature") == amt_feature and cl.get("type") == "range":
                thr = cl.get("low") if cl.get("low") is not None else cl.get("high")
                break
        return (-c["votes"], float(thr) if thr is not None else 0.0)

    merged.sort(key=sort_key)
    return merged

def evaluate_rules(rules, X_val, y_val, min_support=20):
    results = []
    total_fraud = y_val.sum()

    for rule in rules:
        mask = pd.Series(True, index=X_val.index)
        for clause in rule['clauses']:
            feat = clause['feature']
            if clause['type'] == 'eq':
                mask &= (X_val[feat] == clause['value'])
            elif clause['type'] == 'range':
                if clause['low'] is not None:
                    mask &= (X_val[feat] > clause['low'])
                if clause['high'] is not None:
                    mask &= (X_val[feat] <= clause['high'])

        matched_idx = X_val[mask].index
        support = len(matched_idx)
        if support < min_support:  # too few to trust
            continue
        
        tp = y_val.loc[matched_idx].sum()
        precision = tp / support if support > 0 else 0
        recall = tp / total_fraud if total_fraud > 0 else 0

        ci_low, ci_high = proportion_confint(tp, support, alpha=0.05, method="wilson")
        
        results.append({
            "rule": rule['clauses'],
            "votes": rule['votes'],
            "support": support,
            "precision": precision,
            "recall": recall,
            "precision_CI": (ci_low, ci_high)
        })
    
    return pd.DataFrame(results).sort_values("precision", ascending=False)

import math
from collections import defaultdict

# --- Configure your one-hot groups here ---
# prefix -> original column name
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

def _sql_ident(name: str) -> str:
    # Your column names look safe; quote if you need to handle spaces/mixed case.
    return name

def _sql_val(v):
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        # keep integers pretty
        return str(int(v)) if float(v).is_integer() else str(v)
    return f"'{str(v)}'"

def _prefix_for_feature(feat: str):
    for pref in GROUP_MAP.keys():
        if feat.startswith(pref):
            return pref
    return None

def _extract_cat_value(prefix: str, feat: str) -> str:
    # strip "ProductCD_" from "ProductCD_R" -> "R"
    return feat[len(prefix):]

def _range_to_sql(feature, low, low_incl, high, high_incl):
    parts = []
    if low is not None:
        parts.append(f"{_sql_ident(feature)} {'>=' if low_incl else '>'} {_sql_val(low)}")
    if high is not None:
        parts.append(f"{_sql_ident(feature)} {'<=' if high_incl else '<'} {_sql_val(high)}")
    return " AND ".join(parts) if parts else "1=1"

def clauses_to_sql_where_pretty(clauses: list, group_map=GROUP_MAP, group_levels=GROUP_LEVELS):
    """
    Convert your mined rule 'clauses' (eq, range) into a single SQL WHERE string,
    collapsing one-hot columns back to original categoricals.

    Returns: (sql_where_str, english_summary_str)
    """
    # 1) Collect categorical info by group
    pos_vals = defaultdict(set)   # original_col -> set of positive category values
    neg_vals = defaultdict(set)   # original_col -> set of negative category values

    non_cat_preds = []            # numeric/range or non-onehot eqs (kept verbatim)

    for c in clauses:
        t = c.get("type")
        feat = c.get("feature")

        pref = _prefix_for_feature(feat)
        if pref is not None and t == "eq":
            orig_col = group_map[pref]
            val = _extract_cat_value(pref, feat)
            if c.get("value") == 1:
                pos_vals[orig_col].add(val)
            else:
                neg_vals[orig_col].add(val)
            continue

        if t == "eq":
            non_cat_preds.append(f"{_sql_ident(feat)} = {_sql_val(c['value'])}")
        elif t == "range":
            non_cat_preds.append(_range_to_sql(
                feat,
                c.get("low"), c.get("low_inclusive", False),
                c.get("high"), c.get("high_inclusive", False)
            ))
        else:
            non_cat_preds.append("1=1")

    # 2) Build categorical predicates (prefer positives; fall back to NOT IN if only negatives)
    cat_preds = []
    cat_english = []

    all_cat_cols = set(pos_vals.keys()) | set(neg_vals.keys())
    for col in sorted(all_cat_cols):
        positives = sorted(pos_vals[col])
        negatives = sorted(neg_vals[col])

        if positives:
            if len(positives) == 1:
                cat_preds.append(f"{_sql_ident(col)} = {_sql_val(positives[0])}")
                cat_english.append(f"{col} = {positives[0]}")
            else:
                in_list = ", ".join(_sql_val(v) for v in positives)
                cat_preds.append(f"{_sql_ident(col)} IN ({in_list})")
                cat_english.append(f"{col} in ({', '.join(positives)})")
        else:
            # only negatives present; emit NOT IN if we know all levels
            all_levels = group_levels.get([k for k, v in GROUP_MAP.items() if v == col][0], None)
            if all_levels:
                keep = [lvl for lvl in all_levels if lvl not in negatives]
                if len(keep) == 1:
                    cat_preds.append(f"{_sql_ident(col)} = {_sql_val(keep[0])}")
                    cat_english.append(f"{col} = {keep[0]}")
                else:
                    in_list = ", ".join(_sql_val(v) for v in keep)
                    cat_preds.append(f"{_sql_ident(col)} IN ({in_list})")
                    cat_english.append(f"{col} in ({', '.join(keep)})")
            else:
                # If we don't know all levels, safest is to skip emitting a NOT IN
                # (or emit explicit NOT IN on observed negatives)
                notin_list = ", ".join(_sql_val(v) for v in negatives)
                cat_preds.append(f"{_sql_ident(col)} NOT IN ({notin_list})")
                cat_english.append(f"{col} not in ({', '.join(negatives)})")

    # 3) Combine everything
    preds = [p for p in (cat_preds + non_cat_preds) if p and p != "1=1"]
    sql_where = "(" + " AND ".join(preds) + ")" if preds else "(1=1)"

    # English summary (nice for your catalog)
    english_bits = []
    english_bits += cat_english
    for c in clauses:
        if c.get("type") == "range":
            f = c["feature"]
            if f in GROUP_MAP.values():  # skip if we accidentally pass cat column here
                continue
            lo, hi = c.get("low"), c.get("high")
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
