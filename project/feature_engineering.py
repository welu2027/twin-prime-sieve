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

    # residue classes — mod210 and mod2310 are Prime Generator classes
    df["mod6"]    = df["p"] % 6
    df["mod30"]   = df["p"] % 30
    df["mod210"]  = df["p"] % 210
    df["mod2310"] = df["p"] % 2310

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

    # rolling stats of past gaps (shift by 1 so we only see past, not current row)
    gap_series = pd.Series(gap_after, dtype=np.float64)
    df["rolling_mean_gap"] = gap_series.shift(1).rolling(window=100, min_periods=1).mean().values
    df["rolling_std_gap"]  = gap_series.shift(1).rolling(window=100, min_periods=1).std().fillna(0).values

    # normalized index
    df["index_norm"] = df["index"].values / (n - 1) if n > 1 else 0.0

    # Hardy-Littlewood prediction for local density: 2*C2 / (log p)^2
    # pi_2(x) ~ 2*C2 * x / (ln x)^2, so density = 2*C2 / (ln p)^2
    C2 = 0.6601618158468696
    df["hl_density"] = 2 * C2 / (np.log(p) ** 2)

    # residual from H-L prediction (deviation)
    df["density_residual"] = df["local_density"] - df["hl_density"]

    # ratio of actual gap to H-L expected gap (1/hl_density = log_p^2/C2)
    # hl_ratio ≈ 1 means H-L is correct; >1 means larger gap than expected
    expected_hl_gap = 1.0 / df["hl_density"].values
    df["hl_ratio"] = np.where(expected_hl_gap > 0, gap_after / expected_hl_gap, np.nan)

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
