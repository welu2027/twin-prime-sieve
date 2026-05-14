"""
Symbolic regression with PySR to find formulas for twin prime gap deviations
from the Hardy-Littlewood conjecture.

Target: hl_ratio = gap_after / expected_HL_gap
  - hl_ratio = 1  →  gap exactly matches H-L prediction
  - hl_ratio > 1  →  gap larger than H-L predicts
  - hl_ratio < 1  →  gap smaller than H-L predicts

Two modes:
  Default:          features = log_p, mod210, mod2310, local_density
  --focus-residues: features = log_p, mod210, mod2310  (no local_density)
                    Forces PySR to find residue class corrections directly.
                    Use this after confirming residue deviations in analyze_results.py.

Saves equations to data/pysr_equations.csv (default) or
data/pysr_residue_equations.csv (--focus-residues).
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor

from utils import DATA_DIR

DEFAULT_FEATURES  = ["log_p", "mod210", "mod2310", "local_density"]
RESIDUE_FEATURES  = ["log_p", "mod210", "mod2310"]
TARGET = "hl_ratio"


def write_run_notes(path: Path, args, feature_cols: list[str]):
    """Write a human-readable explanation of why this run was done."""
    if args.focus_residues:
        reason = (
            "RERUN REASON: Focus on residue class corrections\n"
            "\n"
            "The initial PySR run (with local_density as a feature) found only a\n"
            "local-density correction: hl_ratio ≈ 1.27 - log(local_density) - log(log_p²).\n"
            "This is mathematically equivalent to H-L and does not improve on it.\n"
            "\n"
            "However, analyze_results.py revealed systematic ~4-5% deviations from\n"
            "H-L in specific mod210 residue classes:\n"
            "  - p ≡ 29, 41, 59 mod 210: gaps ~4-5% LARGER than H-L predicts\n"
            "  - p ≡ 149, 167, 179 mod 210: gaps ~4-5% SMALLER than H-L predicts\n"
            "\n"
            "local_density was dominating the PySR search and masking this signal.\n"
            "This run removes local_density so PySR is forced to find residue-based\n"
            "corrections directly. If it finds a clean formula involving mod210 or\n"
            "mod2310, that would be a genuine improvement over Hardy-Littlewood.\n"
        )
    else:
        reason = "Standard run: search for hl_ratio corrections using all features.\n"

    with open(path, "w") as f:
        f.write(f"PySR Symbolic Regression Run\n")
        f.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode      : {'residue-focus (no local_density)' if args.focus_residues else 'standard'}\n")
        f.write(f"Features  : {feature_cols}\n")
        f.write(f"Target    : {TARGET}\n")
        f.write(f"Sample    : {args.sample:,}\n")
        f.write(f"Iterations: {args.niterations}\n")
        f.write(f"\n{'='*60}\n")
        f.write(reason)
        f.write(f"{'='*60}\n\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    parser.add_argument("--niterations", type=int, default=60)
    parser.add_argument("--sample", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-residues", action="store_true",
                        help="Drop local_density; force PySR to find mod210/mod2310 corrections")
    args = parser.parse_args()

    feature_cols = RESIDUE_FEATURES if args.focus_residues else DEFAULT_FEATURES
    out_csv = DATA_DIR / ("pysr_residue_equations.csv" if args.focus_residues else "pysr_equations.csv")
    out_txt = DATA_DIR / ("pysr_residue_run.txt" if args.focus_residues else "pysr_run.txt")

    notes_path = write_run_notes(out_txt, args, feature_cols)
    print(f"Run notes → {notes_path}")

    df = pd.read_parquet(args.inp)
    df = df.dropna(subset=feature_cols + [TARGET])
    df = df[(df[TARGET] > 0) & (df[TARGET] < 20) & (df["log_p"] > 5)]

    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed)
    print(f"Running symbolic regression on {len(df):,} samples.")
    print(f"Features: {feature_cols}")
    print(f"Target  : {TARGET}\n")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    model = PySRRegressor(
        niterations=args.niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "square"],
        populations=20,
        population_size=50,
        maxsize=10,
        parsimony=0.005,
        batching=True,
        batch_size=10000,
        random_state=args.seed,
        verbosity=1,
        temp_equation_file=True,
    )
    model.fit(X, y, variable_names=feature_cols)

    model.equations_.to_csv(out_csv, index=False)

    result_lines = [
        "\n" + "="*60,
        "RESULTS",
        "="*60,
        model.equations_[["equation", "loss", "complexity"]].to_string(index=False),
        f"\nBest equation: {model.sympy()}",
    ]
    result_text = "\n".join(result_lines)
    print(result_text)

    with open(out_txt, "a") as f:
        f.write(result_text + "\n")

    print(f"\nEquations → {out_csv}")
    print(f"Full notes → {out_txt}")


if __name__ == "__main__":
    main()
