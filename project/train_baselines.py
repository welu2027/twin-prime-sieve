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

from utils import DATA_DIR, MODELS_DIR

FEATURE_COLS = [
    "log_p", "gap_before", "mod30", "mod210", "mod2310",
    "local_density", "hl_density", "log_gap_ratio",
    "rolling_mean_gap", "rolling_std_gap", "index_norm",
]
TARGET = "gap_after"


def evaluate(name: str, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[{name}] MAE={mae:.2f}  R²={r2:.4f}")
    return {"name": name, "mae": mae, "r2": r2}


def naive_baselines(y_train, y_test, gap_before_test):
    """Print MAE/R² for trivial predictors so ML results can be compared honestly."""
    mean_pred = np.full_like(y_test, y_train.mean())
    prev_pred  = gap_before_test

    for name, preds in [("Naive-Mean", mean_pred), ("Naive-PrevGap", prev_pred)]:
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        print(f"[{name:20s}] MAE={mae:.2f}  R²={r2:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=500_000,
                        help="Cap training rows (keeps eval fast; 500K is plenty for honest R²)")
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df = df[df[TARGET] > 0]
    if len(df) > args.max_train:
        df = df.sample(args.max_train, random_state=args.seed)
        print(f"Sampled {args.max_train:,} rows for training (full dataset: {len(df):,})")
    print(f"Training on {len(df):,} samples.")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    gap_before_all = df["gap_before"].values.astype(np.float32)

    X_train, X_test, y_train, y_test, _, gb_test = train_test_split(
        X, y, gap_before_all, test_size=args.test_size, random_state=args.seed
    )

    print("--- Naive baselines ---")
    naive_baselines(y_train, y_test, gb_test)
    print("--- ML models ---")

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
        out = MODELS_DIR / f"model_{name.lower()}.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved to {out}")

    print("\nSummary:")
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"  {r['name']:20s}  MAE={r['mae']:.2f}  R²={r['r2']:.4f}")


if __name__ == "__main__":
    main()
