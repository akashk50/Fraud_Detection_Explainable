import numpy as np


# Split fraud and nonfraud in a time series safe way

def sampleSplit(df, n_bootstraps, TIME_COL, TARGET):
  df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
  cut = int(len(df_sorted) * 0.8)
  train = df_sorted.iloc[:cut].copy()
  val   = df_sorted.iloc[cut:].copy()

  X_train = train.drop(columns=[TARGET])
  y_train = train[TARGET].astype(int)

  X_val = val.drop(columns=[TARGET])
  y_val = val[TARGET].astype(int)

  feature_names = X_train.columns.to_list()
  print(feature_names)

  # -----------------------------
  # Stratified bootstrap sampler
  # -----------------------------
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