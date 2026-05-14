"""
Symbolic regression with PySR to discover mathematical formulas for twin prime gaps.

Tries to find a closed-form expression that predicts gap_after from log_p
and other features. This is the most interesting part of the project — the AI
may rediscover Hardy-Littlewood or find new patterns.

Saves discovered equations to data/pysr_equations.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor

from utils import DATA_DIR

FEATURE_COLS = ["log_p", "gap_before", "local_density", "hl_density"]
TARGET = "gap_after"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--niterations", type=int, default=40,
                        help="PySR evolution iterations (more = better but slower)")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Number of rows to sample for symbolic regression")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.inp)
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df = df[df[TARGET] > 0]

    # subsample — PySR is slow on large datasets
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed)
    print(f"Running symbolic regression on {len(df):,} samples.")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    model = PySRRegressor(
        niterations=args.niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "exp", "square"],
        populations=15,
        population_size=33,
        maxsize=20,
        parsimony=0.001,
        random_state=args.seed,
        verbosity=1,
        equation_file=str(DATA_DIR / "pysr_equations.csv"),
    )
    model.fit(X, y, variable_names=FEATURE_COLS)

    print("\nTop equations discovered:")
    print(model.equations_[["equation", "loss", "complexity"]].head(10).to_string())
    print(f"\nBest equation: {model.sympy()}")


if __name__ == "__main__":
    main()
