"""
Analysis and visualization of twin prime data + model results.

Produces plots saved to data/:
  - twin_prime_density.png     : actual vs Hardy-Littlewood predicted density
  - gap_distribution.png       : histogram of twin prime gaps
  - model_comparison.png       : prediction scatter plots for each model
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import DATA_DIR

FEATURE_COLS = [
    "log_p", "gap_before", "mod30", "mod210",
    "local_density", "hl_density", "log_gap_ratio", "index_norm",
]
TARGET = "gap_after"
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (9, 5)})


def plot_density(df: pd.DataFrame):
    fig, ax = plt.subplots()
    sample = df.sample(min(10_000, len(df)), random_state=42).sort_values("p")
    ax.scatter(np.log10(sample["p"]), sample["local_density"],
               s=4, alpha=0.4, label="Observed density")
    ax.plot(np.log10(sample["p"]), sample["hl_density"],
            color="red", linewidth=1.5, label="Hardy-Littlewood $C_2/(\\ln p)^2$")
    ax.set_xlabel("$\\log_{10}(p)$")
    ax.set_ylabel("Local twin prime density")
    ax.set_title("Twin Prime Density vs Hardy-Littlewood Prediction")
    ax.legend()
    out = DATA_DIR / "twin_prime_density.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_gap_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots()
    gaps = df["gap_after"].values
    gaps = gaps[gaps > 0]
    ax.hist(gaps, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Gap to next twin prime pair")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Twin Prime Gaps")
    out = DATA_DIR / "gap_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_model_comparison(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df = df[df[TARGET] > 0]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_files = list(DATA_DIR.glob("model_*.pkl"))
    if not model_files:
        print("No trained baseline models found. Run train_baselines.py first.")
        return

    fig, axes = plt.subplots(1, len(model_files), figsize=(6 * len(model_files), 5))
    if len(model_files) == 1:
        axes = [axes]

    for ax, mf in zip(axes, model_files):
        with open(mf, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        ax.scatter(y_test, preds, s=3, alpha=0.3)
        lim = max(y_test.max(), preds.max())
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5)
        ax.set_xlabel("Actual gap")
        ax.set_ylabel("Predicted gap")
        ax.set_title(f"{mf.stem}\nMAE={mae:.1f}, R²={r2:.3f}")

    fig.tight_layout()
    out = DATA_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"Loaded {len(df):,} rows.")

    plot_density(df)
    plot_gap_distribution(df)
    plot_model_comparison(df)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
