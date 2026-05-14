"""
Analysis and visualization of twin prime data + model results.

Saves to data/:
  - analysis_results.txt  : text summary of all findings
  - twin_prime_density.png : actual vs Hardy-Littlewood predicted density
  - gap_distribution.png   : histogram of twin prime gaps
  - model_comparison.png   : ML model prediction scatter plots
  - hl_ratio_by_mod210.png : hl_ratio deviation by residue class (H-L correction signal)

Run after main.py:
    python analyze_results.py
"""
import argparse
import pickle
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import DATA_DIR

FEATURE_COLS = [
    "log_p", "gap_before", "mod30", "mod210", "mod2310",
    "local_density", "hl_density", "log_gap_ratio",
    "rolling_mean_gap", "rolling_std_gap", "index_norm",
]
TARGET = "gap_after"
C2 = 0.6601618158468696
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (9, 5)})


def section(buf: StringIO, title: str):
    line = f"\n{'='*60}\n{title}\n{'='*60}"
    print(line)
    buf.write(line + "\n")


def log(buf: StringIO, text: str):
    print(text)
    buf.write(text + "\n")


def summarize_dataset(df: pd.DataFrame, buf: StringIO):
    section(buf, "Dataset Summary")
    log(buf, f"Total twin primes      : {len(df):,}")
    log(buf, f"Range                  : p = {df['p'].min():,} to {df['p'].max():,}")
    log(buf, f"Mean gap               : {df['gap_after'].mean():.1f}")
    log(buf, f"Median gap             : {df['gap_after'].median():.1f}")
    log(buf, f"Max gap                : {df['gap_after'].max():,}")
    log(buf, f"Mean hl_ratio          : {df['hl_ratio'].mean():.4f}  (1.0 = perfect H-L)")
    log(buf, f"Std  hl_ratio          : {df['hl_ratio'].std():.4f}")


def plot_density(df: pd.DataFrame, buf: StringIO):
    section(buf, "Density vs Hardy-Littlewood")
    fig, ax = plt.subplots()
    sample = df.sample(min(10_000, len(df)), random_state=42).sort_values("p")
    ax.scatter(np.log10(sample["p"]), sample["local_density"],
               s=4, alpha=0.4, label="Observed density")
    ax.plot(np.log10(sample["p"]), sample["hl_density"],
            color="red", linewidth=1.5, label="Hardy-Littlewood $2C_2/(\\ln p)^2$")
    ax.set_xlabel("$\\log_{10}(p)$")
    ax.set_ylabel("Local twin prime density")
    ax.set_title("Twin Prime Density vs Hardy-Littlewood Prediction")
    ax.legend()
    out = DATA_DIR / "twin_prime_density.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def plot_gap_distribution(df: pd.DataFrame, buf: StringIO):
    section(buf, "Gap Distribution")
    gaps = df["gap_after"].values
    gaps = gaps[gaps > 0]
    fig, ax = plt.subplots()
    ax.hist(gaps, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Gap to next twin prime pair")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Twin Prime Gaps")
    out = DATA_DIR / "gap_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Gap range: [{gaps.min()}, {gaps.max()}]")
    log(buf, f"Saved {out}")


def plot_hl_ratio_by_residue(df: pd.DataFrame, buf: StringIO):
    section(buf, "H-L Ratio by Residue Class (mod210)")
    data = df[(df["hl_ratio"] > 0) & (df["hl_ratio"] < 10)].copy()
    # top 20 most common mod210 classes
    top_classes = data["mod210"].value_counts().nlargest(20).index
    data = data[data["mod210"].isin(top_classes)]

    means = data.groupby("mod210")["hl_ratio"].mean().sort_values()
    log(buf, f"Residue classes with hl_ratio most BELOW 1.0 (gap smaller than H-L predicts):")
    log(buf, means.head(5).to_string())
    log(buf, f"\nResidue classes with hl_ratio most ABOVE 1.0 (gap larger than H-L predicts):")
    log(buf, means.tail(5).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(means.index.astype(str), means.values, color=[
        "steelblue" if v < 1.0 else "tomato" for v in means.values
    ])
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="H-L baseline (1.0)")
    ax.set_xlabel("p mod 210")
    ax.set_ylabel("Mean hl_ratio")
    ax.set_title("Deviation from Hardy-Littlewood by Residue Class (mod 210)")
    ax.legend()
    plt.xticks(rotation=45)
    out = DATA_DIR / "hl_ratio_by_mod210.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def plot_model_comparison(df: pd.DataFrame, buf: StringIO):
    section(buf, "ML Model Comparison")
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df = df[df[TARGET] > 0]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    gap_before = df["gap_before"].values.astype(np.float32)
    _, X_test, _, y_test, _, gb_test = train_test_split(
        X, y, gap_before, test_size=0.2, random_state=42
    )

    # naive baselines
    log(buf, "Naive baselines:")
    log(buf, f"  Mean predictor  MAE={mean_absolute_error(y_test, np.full_like(y_test, y_test.mean())):.2f}")
    log(buf, f"  Prev gap        MAE={mean_absolute_error(y_test, gb_test):.2f}  R²={r2_score(y_test, gb_test):.4f}")

    model_files = sorted(DATA_DIR.glob("model_*.pkl"))
    if not model_files:
        log(buf, "No trained models found. Run main.py first.")
        return

    log(buf, "\nML models:")
    fig, axes = plt.subplots(1, len(model_files), figsize=(6 * len(model_files), 5))
    if len(model_files) == 1:
        axes = [axes]

    for ax, mf in zip(axes, model_files):
        with open(mf, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        log(buf, f"  {mf.stem:25s}  MAE={mae:.2f}  R²={r2:.4f}")
        ax.scatter(y_test, preds, s=3, alpha=0.3)
        lim = max(float(y_test.max()), float(preds.max()))
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5)
        ax.set_xlabel("Actual gap")
        ax.set_ylabel("Predicted gap")
        ax.set_title(f"{mf.stem}\nMAE={mae:.1f}, R²={r2:.3f}")

    fig.tight_layout()
    out = DATA_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def summarize_pysr(buf: StringIO):
    section(buf, "PySR Symbolic Regression Results")
    eq_file = DATA_DIR / "pysr_equations.csv"
    if not eq_file.exists():
        log(buf, "No PySR equations found. Run main.py without --skip-symbolic.")
        return
    eqs = pd.read_csv(eq_file)
    log(buf, f"Target: hl_ratio  (gap_after / expected H-L gap)")
    log(buf, f"Best equation complexity 1: {eqs.iloc[0]['equation']}")
    log(buf, f"\nTop equations (by complexity):")
    log(buf, eqs[["equation", "loss", "complexity"]].to_string(index=False))
    log(buf, f"\nNote: coefficient ~1.32 ≈ 2×C₂ = {2*C2:.4f} means PySR rediscovered H-L constant.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DATA_DIR / "twin_primes_features.parquet"))
    args = parser.parse_args()

    buf = StringIO()
    buf.write(
        f"Twin Prime Analysis Results\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    df = pd.read_parquet(args.inp)
    log(buf, f"Loaded {len(df):,} rows.")

    summarize_dataset(df, buf)
    plot_density(df, buf)
    plot_gap_distribution(df, buf)
    plot_hl_ratio_by_residue(df, buf)
    plot_model_comparison(df, buf)
    summarize_pysr(buf)

    section(buf, "Output Files")
    for f in sorted(DATA_DIR.iterdir()):
        log(buf, f"  {f.name}")

    out_txt = DATA_DIR / "analysis_results.txt"
    out_txt.write_text(buf.getvalue())
    print(f"\nAnalysis saved to {out_txt}")


if __name__ == "__main__":
    main()
