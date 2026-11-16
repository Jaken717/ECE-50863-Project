"""Shared data loading, cleaning and splitting for all experiments.

Label encoding (from the original dataset documentation, verified empirically,
e.g. sslfinal_state=1 "trusted HTTPS" co-occurs with result=1 in 89% of rows):
    result = -1  -> phishing     (positive class, y = 1)
    result =  1  -> legitimate   (negative class, y = 0)

Split: stratified 70/15/15 train/val/test. Rows with identical feature vectors
are kept in the same split (group-aware), because ~47% of rows are exact
duplicates and splitting them across sets would leak test rows into training.

Run directly to (re)create data/splits.csv and print a data-quality summary:
    python src/preprocessing.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "phishing_websites.csv"
SPLITS_CSV = ROOT / "data" / "splits.csv"
TARGET = "result"
RANDOM_STATE = 42


def load_raw() -> pd.DataFrame:
    if not DATA_CSV.exists():
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=327)
        DATA_CSV.parent.mkdir(exist_ok=True)
        ds.data.features.join(ds.data.targets).to_csv(DATA_CSV, index=False)
    return pd.read_csv(DATA_CSV)


def feature_groups(X: pd.DataFrame) -> np.ndarray:
    """Integer id per distinct feature vector (identical rows share an id)."""
    return X.groupby(list(X.columns), sort=False).ngroup().to_numpy()


def quality_report(df: pd.DataFrame) -> dict:
    X = df.drop(columns=TARGET)
    labels_per_vec = df.groupby(list(X.columns))[TARGET].nunique()
    return {
        "rows": len(df),
        "features": X.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_vectors": int(X.duplicated().sum()),
        "unique_feature_vectors": int(len(labels_per_vec)),
        "conflicting_vectors": int((labels_per_vec > 1).sum()),
        "phishing": int((df[TARGET] == -1).sum()),
        "legitimate": int((df[TARGET] == 1).sum()),
    }


def make_splits(df: pd.DataFrame) -> pd.Series:
    """Assign each row to train/val/test (≈70/15/15), stratified and group-aware.

    20 stratified group folds of ≈5% each: 3 -> test, 3 -> val, 14 -> train.
    """
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == -1).astype(int).to_numpy()
    groups = feature_groups(X)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=RANDOM_STATE)
    split = np.empty(len(df), dtype=object)
    for fold, (_, idx) in enumerate(sgkf.split(X, y, groups)):
        split[idx] = "test" if fold < 3 else "val" if fold < 6 else "train"
    return pd.Series(split, index=df.index, name="split")


def load_splits():
    """Return dict with X_train, y_train, X_val, y_val, X_test, y_test.

    Duplicates and conflicting rows are kept (they reflect the real data), but
    identical vectors never cross splits. y is 1 = phishing, 0 = legitimate.
    """
    df = load_raw()
    if SPLITS_CSV.exists():
        split = pd.read_csv(SPLITS_CSV)["split"]
        assert len(split) == len(df), "splits.csv does not match dataset; delete it"
    else:
        split = make_splits(df)
        split.to_csv(SPLITS_CSV, index=False)

    X = df.drop(columns=TARGET)
    y = (df[TARGET] == -1).astype(int).rename("phishing")
    out = {}
    for name in ("train", "val", "test"):
        m = (split == name).to_numpy()
        out[f"X_{name}"], out[f"y_{name}"] = X[m].reset_index(drop=True), y[m].reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_raw()
    for k, v in quality_report(df).items():
        print(f"{k:>27}: {v}")
    if SPLITS_CSV.exists():
        SPLITS_CSV.unlink()
    s = load_splits()
    print("\nsplit sizes / phishing rate:")
    for name in ("train", "val", "test"):
        y = s[f"y_{name}"]
        print(f"  {name:>5}: {len(y):5d} rows ({len(y)/len(df):.1%}), phishing {y.mean():.1%}")
    # leakage check: no feature vector appears in two splits
    keys = {n: set(map(tuple, s[f"X_{n}"].to_numpy())) for n in ("train", "val", "test")}
    print("  overlap train∩test:", len(keys["train"] & keys["test"]),
          " train∩val:", len(keys["train"] & keys["val"]),
          " val∩test:", len(keys["val"] & keys["test"]))
