"""
Symbolic regression with PySR to find formulas for twin prime gap deviations
from the Hardy-Littlewood conjecture.

Target: hl_ratio = gap_after / expected_HL_gap
  - hl_ratio = 1  →  gap exactly matches H-L prediction
  - hl_ratio > 1  →  gap larger than H-L predicts
  - hl_ratio < 1  →  gap smaller than H-L predicts

Goal: find a clean formula for hl_ratio using Prime Generator residue classes
(mod210, mod2310) and local density, improving on H-L locally.

Saves discovered equations to data/pysr_equations.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor

from utils import DATA_DIR

FEATURE_COLS = ["log_p", "mod210", "mod2310", "local_density"]
TARGET = "hl_ratio"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--niterations", type=int, default=60,
                        help="PySR evolution iterations (more = better but slower)")
    parser.add_argument("--sample", type=int, default=50000,
                        help="Number of rows to sample for symbolic regression")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    # hl_ratio > 0 and reasonable (exclude extreme outliers at very small p)
    df = df[(df[TARGET] > 0) & (df[TARGET] < 20) & (df["log_p"] > 5)]

    # subsample — PySR is slow on large datasets
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed)
    print(f"Running symbolic regression on {len(df):,} samples.")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    model = PySRRegressor(
        niterations=args.niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "square"],
        populations=20,
        population_size=50,
        maxsize=10,        # force clean, interpretable equations
        parsimony=0.005,   # stronger parsimony penalty to prefer simpler formulas
        random_state=args.seed,
        verbosity=1,
        temp_equation_file=True,
    )
    model.fit(X, y, variable_names=FEATURE_COLS)

    out_csv = DATA_DIR / "pysr_equations.csv"
    model.equations_.to_csv(out_csv, index=False)
    print(f"Saved equations to {out_csv}")

    print("\nTop equations discovered:")
    print(model.equations_[["equation", "loss", "complexity"]].head(10).to_string())
    print(f"\nBest equation: {model.sympy()}")


if __name__ == "__main__":
    main()
