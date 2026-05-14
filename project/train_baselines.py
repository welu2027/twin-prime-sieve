"""
Train XGBoost and Random Forest models to predict gap_after from features.

Target: gap_after (gap to the next lower twin prime)
Features: log_p, gap_before, mod30, mod210, local_density, hl_density, log_gap_ratio

Saves trained models to data/ and prints evaluation metrics.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from utils import DATA_DIR

FEATURE_COLS = [
    "log_p", "gap_before", "mod30", "mod210",
    "local_density", "hl_density", "log_gap_ratio",
    "index_norm",
]
TARGET = "gap_after"


def evaluate(name: str, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[{name}] MAE={mae:.2f}  R²={r2:.4f}")
    return {"name": name, "mae": mae, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    # drop last row (gap_after is 0 sentinel)
    df = df[df[TARGET] > 0]
    print(f"Training on {len(df):,} samples.")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=args.seed, verbosity=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=args.seed
        ),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        out = DATA_DIR / f"model_{name.lower()}.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved to {out}")

    print("\nSummary:")
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"  {r['name']:20s}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}")


if __name__ == "__main__":
    main()
