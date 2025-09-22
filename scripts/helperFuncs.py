# helperFuncs.py
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd


# ---- add to decisionTree.py (top of file) ----
import numpy as np  # you’ll need this later for bootstraps

# --- DFS over sklearn tree -> raw paths --------------------------------------
def dfs_leaf_paths(
    estimator,
    feature_names: List[str],
    positive_class: int = 1,
    keep_all: bool = False
) -> List[Dict[str, Any]]:
    tree = estimator.tree_
    children_left  = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value  # (n_nodes, 1, n_classes)

    def is_leaf(node_id: int) -> bool:
        return children_left[node_id] == children_right[node_id]

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
    stack: List[tuple[int, List[Dict[str, Any]]]] = [(0, [])]  # (node_id, path_so_far)

    while stack:
        node_id, path_so_far = stack.pop()

        if is_leaf(node_id):
            meta = leaf_meta(node_id)
            if keep_all or meta["predicted_class"] == positive_class:
                paths.append({
                    "leaf_id": node_id,
                    "path": path_so_far,
                    **meta,
                })
            continue

        f_idx = features[node_id]
        thr = thresholds[node_id]
        if f_idx < 0 or f_idx >= len(feature_names):
            if children_right[node_id] != -1:
                stack.append((children_right[node_id], list(path_so_far)))
            if children_left[node_id] != -1:
                stack.append((children_left[node_id], list(path_so_far)))
            continue

        f_name = feature_names[f_idx]
        left_clause = {"feature": f_name, "op": "<=", "threshold": float(thr), "feature_idx": int(f_idx)}
        right_clause = {"feature": f_name, "op": ">",  "threshold": float(thr), "feature_idx": int(f_idx)}

        if children_right[node_id] != -1:
            stack.append((children_right[node_id], path_so_far + [right_clause]))
        if children_left[node_id] != -1:
            stack.append((children_left[node_id],  path_so_far + [left_clause]))

    return paths

# --- Normalization (no snapping) ---------------------------------------------
INTEGER_LIKE_FEATURES = {"hour_of_day", "day_of_week"}

def is_one_hot_split(_feature_name: str, threshold: float) -> bool:
    return 0.0 < threshold < 1.0  # heuristic

def clean_and_merge(
    raw_path: List[Dict[str, Any]],
    integer_like: set = INTEGER_LIKE_FEATURES
) -> List[Dict[str, Any]] | None:
    """
    Normalize a root→leaf path into simple clauses WITHOUT snapping.
    - One-hot near 0.5 -> equality on 0/1
    - Integer-like -> inclusive integer bounds
    - Continuous -> keep raw thresholds
    - Merge repeated constraints; drop contradictions
    """
    eq_constraints: Dict[str, int] = {}
    int_ranges: Dict[str, Dict[str, int]] = {}
    cont_ranges: Dict[str, Dict[str, Any]] = {}

    def add_one_hot_eq(feat: str, val: int) -> bool:
        prev = eq_constraints.get(feat)
        if prev is not None and prev != val:
            return False
        eq_constraints[feat] = val
        return True

    def add_int_bound(feat: str, op: str, thr: float) -> bool:
        if op == "<=":
            hi = int(np.floor(thr))
            rng = int_ranges.setdefault(feat, {"low": None, "high": None})
            rng["high"] = hi if rng["high"] is None else min(rng["high"], hi)
        else:
            lo = int(np.ceil(thr))
            rng = int_ranges.setdefault(feat, {"low": None, "high": None})
            rng["low"] = lo if rng["low"] is None else max(rng["low"], lo)
        return True

    def add_cont_bound(feat: str, op: str, thr: float) -> bool:
        rng = cont_ranges.setdefault(feat, {"low": None, "low_inc": False, "high": None, "high_inc": False})
        if op == "<=":
            if rng["high"] is None or thr < rng["high"]:
                rng["high"] = float(thr); rng["high_inc"] = True
            elif thr == rng["high"]:
                rng["high_inc"] = True
        else:
            if rng["low"] is None or thr > rng["low"]:
                rng["low"] = float(thr); rng["low_inc"] = False
            elif thr == rng["low"]:
                rng["low_inc"] = rng["low_inc"] and False
        return True

    for clause in raw_path:
        feat, op, thr = clause["feature"], clause["op"], float(clause["threshold"])
        if is_one_hot_split(feat, thr):
            val = 0 if op == "<=" else 1
            if not add_one_hot_eq(feat, val): return None
            continue
        if feat in integer_like:
            if not add_int_bound(feat, op, thr): return None
            continue
        if not add_cont_bound(feat, op, thr): return None

    for feat, rng in int_ranges.items():
        lo, hi = rng["low"], rng["high"]
        if lo is not None and hi is not None and lo > hi: return None
    for feat, rng in cont_ranges.items():
        lo, hi = rng["low"], rng["high"]
        if lo is not None and hi is not None:
            if lo > hi: return None
            if lo == hi and (not rng["low_inc"] or not rng["high_inc"]): return None

    norm: List[Dict[str, Any]] = []
    for feat, val in eq_constraints.items():
        norm.append({"feature": feat, "type": "eq", "value": int(val)})
    for feat, rng in int_ranges.items():
        low = None if rng["low"]  is None else int(rng["low"])
        high= None if rng["high"] is None else int(rng["high"])
        if low is None and high is None: continue
        norm.append({"feature": feat, "type": "int_range", "low": low, "high": high})
    for feat, rng in cont_ranges.items():
        lo, hi = rng["low"], rng["high"]
        li, hi_incl = rng["low_inc"], rng["high_inc"]
        if lo is None and hi is None: continue
        norm.append({"feature": feat, "type": "range",
                     "low": lo, "low_inclusive": bool(li) if lo is not None else False,
                     "high": hi, "high_inclusive": bool(hi_incl) if hi is not None else False})

    if not norm: return None
    norm.sort(key=lambda d: (d["feature"], d["type"]))
    return norm

def canonical_key(norm_clauses: List[Dict[str, Any]]) -> Tuple:
    parts = []
    for c in norm_clauses:
        t, f = c["type"], c["feature"]
        if t == "eq":
            parts.append((f, "eq", int(c["value"])))
        elif t == "int_range":
            parts.append((f, "int_range", c.get("low"), c.get("high")))
        elif t == "range":
            parts.append((f, "range", c.get("low"), c.get("low_inclusive", False),
                          c.get("high"), c.get("high_inclusive", False)))
        else:
            parts.append((f, t))
    parts.sort()
    return tuple(parts)

# --- Simple post-processing (no snapping) ------------------------------------
def postprocess_rules(candidates, *, min_votes=5):
    """
    Keep rules with at least `min_votes`. No threshold averaging/snapping.
    """
    cleaned = [ {"votes": int(c["votes"]), "clauses": c["clauses"]}
                for c in candidates if int(c.get("votes", 0)) >= min_votes ]
    cleaned.sort(key=lambda d: d["votes"], reverse=True)
    return cleaned

# --- Evaluation + SQL/English rendering --------------------------------------
from statsmodels.stats.proportion import proportion_confint
import math

def evaluate_rules(rules, X_val: pd.DataFrame, y_val: pd.Series, min_support=20):
    results = []
    total_fraud = int(y_val.sum())

    for rule in rules:
        mask = pd.Series(True, index=X_val.index)
        for c in rule["clauses"]:
            feat = c["feature"]
            if c["type"] == "eq":
                mask &= (X_val[feat] == c["value"])
            elif c["type"] == "int_range":
                if c.get("low")  is not None: mask &= (X_val[feat] >= c["low"])
                if c.get("high") is not None: mask &= (X_val[feat] <= c["high"])
            elif c["type"] == "range":
                if c.get("low")  is not None: mask &= (X_val[feat] >  c["low"])
                if c.get("high") is not None: mask &= (X_val[feat] <= c["high"])

        idx = X_val[mask].index
        support = len(idx)
        if support < min_support: continue

        tp = int(y_val.loc[idx].sum())
        precision = tp / support if support else 0.0
        recall = tp / total_fraud if total_fraud else 0.0
        ci_low, ci_high = proportion_confint(tp, support, alpha=0.05, method="wilson")
        results.append({
            "rule": rule["clauses"],
            "votes": rule["votes"],
            "support": support,
            "precision": precision,
            "recall": recall,
            "precision_CI": (ci_low, ci_high),
        })

    df = pd.DataFrame(results)
    return df.sort_values(["precision","support"], ascending=[False, False]) if not df.empty else df

# categorical collapse helpers (kept from before; works with 'eq' one-hots)
GROUP_MAP = {"ProductCD_": "ProductCD", "card6_": "card6", "card4_": "card4"}
GROUP_LEVELS = {
    "ProductCD_": ["W","R","H","S","C"],
    "card6_": ["credit","debit","charge","debit_or_credit"],
    "card4_": ["visa","mastercard","discover","american_express"],
}

def _sql_ident(name: str) -> str: return name
def _sql_val(v):
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return str(int(v)) if float(v).is_integer() else str(v)
    return f"'{str(v)}'"

def _prefix_for_feature(feat: str):
    for pref in GROUP_MAP: 
        if feat.startswith(pref): return pref
    return None

def _extract_cat_value(prefix: str, feat: str) -> str:
    return feat[len(prefix):]

def _range_to_sql(feature, low, low_incl, high, high_incl):
    parts = []
    if low  is not None: parts.append(f"{_sql_ident(feature)} {'>=' if low_incl else '>'}  {_sql_val(low)}")
    if high is not None: parts.append(f"{_sql_ident(feature)} {'<=' if high_incl else '<'} {_sql_val(high)}")
    return " AND ".join(parts) if parts else "1=1"

def clauses_to_sql_where_pretty(clauses: list, group_map=GROUP_MAP, group_levels=GROUP_LEVELS):
    from collections import defaultdict
    pos_vals, neg_vals = defaultdict(set), defaultdict(set)
    non_cat_preds, cat_preds, cat_english = [], [], []

    for c in clauses:
        t, feat = c.get("type"), c.get("feature")
        pref = _prefix_for_feature(feat)
        if pref and t == "eq":
            orig = group_map[pref]; val = _extract_cat_value(pref, feat)
            (pos_vals if c.get("value")==1 else neg_vals)[orig].add(val)
            continue
        if t == "eq":
            non_cat_preds.append(f"{_sql_ident(feat)} = {_sql_val(c['value'])}")
        elif t == "range":
            non_cat_preds.append(_range_to_sql(feat, c.get("low"), c.get("low_inclusive", False),
                                                    c.get("high"), c.get("high_inclusive", True)))
        elif t == "int_range":
            non_cat_preds.append(_range_to_sql(feat, c.get("low"), True, c.get("high"), True))

    all_cat_cols = set(pos_vals) | set(neg_vals)
    for col in sorted(all_cat_cols):
        positives = sorted(pos_vals[col]); negatives = sorted(neg_vals[col])
        if positives:
            if len(positives)==1:
                cat_preds.append(f"{_sql_ident(col)} = {_sql_val(positives[0])}")
                cat_english.append(f"{col} = {positives[0]}")
            else:
                lst = ", ".join(_sql_val(v) for v in positives)
                cat_preds.append(f"{_sql_ident(col)} IN ({lst})")
                cat_english.append(f"{col} in ({', '.join(positives)})")
        else:
            levels = None
            for pref, oc in GROUP_MAP.items():
                if oc == col: levels = group_levels.get(pref); break
            if levels:
                keep = [lvl for lvl in levels if lvl not in negatives]
                if len(keep)==1:
                    cat_preds.append(f"{_sql_ident(col)} = {_sql_val(keep[0])}")
                    cat_english.append(f"{col} = {keep[0]}")
                else:
                    lst = ", ".join(_sql_val(v) for v in keep)
                    cat_preds.append(f"{_sql_ident(col)} IN ({lst})")
                    cat_english.append(f"{col} in ({', '.join(keep)})")

    preds = [p for p in (cat_preds + non_cat_preds) if p and p != "1=1"]
    sql_where = "(" + " AND ".join(preds) + ")" if preds else "(1=1)"

    english_bits = []
    english_bits += cat_english
    for c in clauses:
        if c.get("type") == "range":
            f, lo, hi = c["feature"], c.get("low"), c.get("high")
            li, hi_incl = c.get("low_inclusive", False), c.get("high_inclusive", True)
            if lo is not None and hi is None: english_bits.append(f"{f} {'≥' if li else '>'} {lo}")
            elif hi is not None and lo is None: english_bits.append(f"{f} {'≤' if hi_incl else '<'} {hi}")
            elif lo is not None and hi is not None: english_bits.append(f"{f} between {lo} and {hi}")
        elif c.get("type") == "int_range":
            f, lo, hi = c["feature"], c.get("low"), c.get("high")
            if lo is not None and hi is not None: english_bits.append(f"{f} in [{lo}, {hi}]")
            elif lo is not None: english_bits.append(f"{f} ≥ {lo}")
            elif hi is not None: english_bits.append(f"{f} ≤ {hi}")

    english = " AND ".join(english_bits) if english_bits else "All transactions"
    return sql_where, english
