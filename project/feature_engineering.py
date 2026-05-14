"""
Feature engineering for the twin prime dataset.

Reads data/twin_primes.parquet and adds features:
  - log_p            : log of lower twin prime
  - gap_before/after : gaps to neighboring twin primes
  - rolling_density  : local twin prime density (twins per unit interval)
  - mod30, mod210    : residue classes
  - gap_ratio        : gap_after / gap_before (pattern signal)
  - prime_index_norm : normalized ordinal index

Saves to data/twin_primes_features.parquet.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_DIR


def add_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    df = df.copy()
    p = df["p"].values.astype(np.float64)
    n = len(df)

    # log scale
    df["log_p"] = np.log(p)

    # gaps
    gap_after = np.empty(n, dtype=np.int64)
    gap_after[:-1] = df["p"].values[1:] - df["p"].values[:-1]
    gap_after[-1] = 0
    df["gap_after"] = gap_after

    # ratio of consecutive gaps (log to avoid inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            df["gap_before"].values > 0,
            np.log(gap_after.astype(float) + 1) - np.log(df["gap_before"].values.astype(float) + 1),
            0.0,
        )
    df["log_gap_ratio"] = ratio

    # residue classes
    df["mod6"]   = df["p"] % 6
    df["mod30"]  = df["p"] % 30
    df["mod210"] = df["p"] % 210

    # local density: twins in a rolling window of `window` consecutive twins
    # approximated as window / (p[i+window/2] - p[i-window/2])
    half = window // 2
    p_int = df["p"].values
    density = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        span = float(p_int[hi] - p_int[lo]) if hi > lo else 1.0
        density[i] = (hi - lo) / span
    df["local_density"] = density

    # normalized index
    df["index_norm"] = df["index"].values / (n - 1) if n > 1 else 0.0

    # Hardy-Littlewood prediction for local density: C2 / (log p)^2
    C2 = 0.6601618158468696
    df["hl_density"] = C2 / (np.log(p) ** 2)

    # residual from H-L prediction (deviation)
    df["density_residual"] = df["local_density"] - df["hl_density"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Add features to twin prime dataset.")
    parser.add_argument("--inp", type=str, default=str(DATA_DIR / "twin_primes.parquet"))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--window", type=int, default=100,
                        help="Rolling window size for local density estimate")
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"Loaded {len(df):,} rows from {args.inp}")

    df = add_features(df, window=args.window)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns to {args.out}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
