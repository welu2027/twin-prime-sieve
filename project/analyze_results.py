"""
Analysis and visualization of twin prime data + model results.

Saves to data/:
  - analysis_results.txt       : text summary of all findings
  - twin_prime_density.png      : actual vs Hardy-Littlewood predicted density
  - gap_distribution.png        : histogram of twin prime gaps
  - model_comparison.png        : ML model prediction scatter plots
  - hl_ratio_by_mod210.png      : hl_ratio deviation by residue class
  - hl_significance_table.png   : t-test significance per class
  - hl_singular_series.png      : correlation with number-theoretic features
  - hl_range_ablation.png       : class deviations across log_p sub-ranges

Run after main.py (and optionally residue_analysis.py):
    python analyze_results.py
"""
import argparse
import pickle
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import DATA_DIR, FIGURES_DIR, MODELS_DIR, RESIDUE_DIR

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
    out = FIGURES_DIR / "twin_prime_density.png"
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
    out = FIGURES_DIR / "gap_distribution.png"
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
    out = FIGURES_DIR / "hl_ratio_by_mod210.png"
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

    model_files = sorted(MODELS_DIR.glob("model_*.pkl"))
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
    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def significance_table(buf: StringIO):
    """Task 1: Full t-test table showing which mod210 classes significantly deviate from H-L."""
    section(buf, "Statistical Significance: Residue Class Deviations from H-L")
    stats_file = RESIDUE_DIR / "residue_class_stats.csv"
    if not stats_file.exists():
        log(buf, "residue_class_stats.csv not found. Run: python residue_analysis.py")
        return

    df = pd.read_csv(stats_file)
    sig = df[df["significant"]].copy()
    below = sig[sig["mean_hl_ratio"] < 1.0].sort_values("mean_hl_ratio")
    above = sig[sig["mean_hl_ratio"] > 1.0].sort_values("mean_hl_ratio", ascending=False)

    log(buf, f"Classes tested  : {len(df)}")
    log(buf, f"Significant (p<0.01): {len(sig)}  [{len(below)} below H-L, {len(above)} above H-L]")
    log(buf, "")
    log(buf, "Full significance table (sorted by mean hl_ratio):")
    cols = ["mod210", "mean_hl_ratio", "std", "n", "t_stat", "p_value", "significant"]
    log(buf, df[cols].sort_values("mean_hl_ratio").to_string(index=False))

    log(buf, "\nMost significant BELOW H-L (gap smaller than predicted):")
    log(buf, below[cols].head(8).to_string(index=False))
    log(buf, "\nMost significant ABOVE H-L (gap larger than predicted):")
    log(buf, above[cols].head(8).to_string(index=False))

    # Bonferroni-corrected threshold
    alpha_bonf = 0.01 / len(df)
    sig_bonf = df[df["p_value"] < alpha_bonf]
    log(buf, f"\nBonferroni-corrected (α={alpha_bonf:.4g}): {len(sig_bonf)} classes remain significant")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_sorted = df.sort_values("mean_hl_ratio")
    colors = ["tomato" if v < 1.0 else "steelblue" for v in df_sorted["mean_hl_ratio"]]
    axes[0].bar(range(len(df_sorted)), df_sorted["mean_hl_ratio"] - 1.0, color=colors)
    axes[0].axhline(0, color="black", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Residue class (sorted by mean hl_ratio)")
    axes[0].set_ylabel("Deviation from H-L  (hl_ratio − 1)")
    axes[0].set_title("Mean hl_ratio Deviation per mod210 Class")

    neg_log_p = -np.log10(df_sorted["p_value"].clip(lower=1e-300))
    sig_line = -np.log10(0.01 / len(df))
    axes[1].bar(range(len(df_sorted)), neg_log_p,
                color=["gold" if v >= sig_line else "lightgray" for v in neg_log_p])
    axes[1].axhline(sig_line, color="red", linewidth=1.5, linestyle="--",
                    label=f"Bonferroni α (−log₁₀={sig_line:.1f})")
    axes[1].set_xlabel("Residue class (sorted by mean hl_ratio)")
    axes[1].set_ylabel("−log₁₀(p-value)")
    axes[1].set_title("T-test Significance per mod210 Class")
    axes[1].legend()

    fig.tight_layout()
    out = FIGURES_DIR / "hl_significance_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def singular_series_comparison(buf: StringIO):
    """Task 2: Show deviations correlate with number-theoretic features beyond H-L."""
    section(buf, "Comparison with H-L Singular Series")
    stats_file = RESIDUE_DIR / "residue_class_stats.csv"
    if not stats_file.exists():
        log(buf, "residue_class_stats.csv not found. Run: python residue_analysis.py")
        return

    df = pd.read_csv(stats_file)
    target = "mean_hl_ratio"
    feat_cols = ["dist_7", "dist_11", "dist_13", "dist_17", "dist_19", "dist_23", "isolation"]

    log(buf, "Under standard H-L, all 48 valid mod210 classes should have hl_ratio = 1.0.")
    log(buf, "If deviations correlate with number-theoretic features of each class, that")
    log(buf, "is evidence the deviations go BEYOND what H-L already accounts for.")
    log(buf, "")
    log(buf, f"{'Feature':<12}  {'Pearson r':>10}  {'p-value':>12}  {'Spearman r':>10}")
    log(buf, "-" * 52)

    best_feat, best_r = None, 0.0
    for feat in feat_cols:
        sub = df[[feat, target]].dropna()
        r_p, p_p = pearsonr(sub[feat], sub[target])
        r_s, p_s = spearmanr(sub[feat], sub[target])
        log(buf, f"{feat:<12}  {r_p:>10.4f}  {p_p:>12.2e}  {r_s:>10.4f}")
        if abs(r_p) > abs(best_r):
            best_r, best_feat = r_p, feat

    log(buf, "")
    log(buf, f"Strongest predictor: {best_feat}  (r={best_r:.4f})")
    log(buf, "")
    log(buf, "Interpretation: correlation with dist_* features means classes 'near' multiples")
    log(buf, "of primes outside the wheel (11, 13, ...) behave differently from H-L.")
    log(buf, "This is a higher-order correction beyond the standard C₂ singular series.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: best feature vs mean_hl_ratio
    x = df[best_feat].values
    y = df[target].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    m, b, r, p, _ = stats.linregress(x, y)
    axes[0].scatter(x, y, color="steelblue", s=60, zorder=3)
    xs = np.linspace(x.min(), x.max(), 200)
    axes[0].plot(xs, m * xs + b, "r--", linewidth=1.5, label=f"r={r:.3f}, p={p:.2e}")
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle=":")
    axes[0].set_xlabel(best_feat)
    axes[0].set_ylabel("Mean hl_ratio")
    axes[0].set_title(f"Mean hl_ratio vs {best_feat}\n(H-L predicts flat line at 1.0)")
    axes[0].legend()

    # Bar: all feature correlations
    rs = [pearsonr(df[f].dropna(), df.loc[df[f].notna(), target])[0] for f in feat_cols]
    colors = ["tomato" if r < 0 else "steelblue" for r in rs]
    axes[1].bar(feat_cols, rs, color=colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Pearson r with mean hl_ratio")
    axes[1].set_title("Feature Correlations with H-L Deviation")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out = FIGURES_DIR / "hl_singular_series.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def range_ablation(df: pd.DataFrame, buf: StringIO):
    """Task 4: Check that residue class deviations are consistent across log_p sub-ranges."""
    section(buf, "Ablation: Class Deviations Across log_p Sub-ranges")
    data = df[(df["hl_ratio"] > 0) & (df["hl_ratio"] < 10)].copy()

    # Define sub-ranges by log_p (natural log)
    log_p_max = data["log_p"].max()
    log_p_min = max(data["log_p"].min(), 5.0)
    edges = np.arange(np.floor(log_p_min), np.ceil(log_p_max) + 1, 1.0)
    ranges = list(zip(edges[:-1], edges[1:]))

    range_means = {}
    log(buf, f"{'Range (log_p)':<22}  {'N twins':>10}  {'Classes w/ data':>16}")
    log(buf, "-" * 54)
    for lo, hi in ranges:
        subset = data[(data["log_p"] >= lo) & (data["log_p"] < hi)]
        if len(subset) < 1000:
            continue
        class_mean = subset.groupby("mod210")["hl_ratio"].mean()
        range_means[f"[{lo:.0f},{hi:.0f})"] = class_mean
        log(buf, f"  [{lo:.0f},{hi:.0f})  (p~10^{lo/np.log(10):.1f}–10^{hi/np.log(10):.1f})  "
               f"{len(subset):>10,}  {len(class_mean):>16}")

    if len(range_means) < 2:
        log(buf, "Not enough sub-ranges with data. Run on larger dataset.")
        return

    pivot = pd.DataFrame(range_means).dropna()
    log(buf, f"\nSpearman rank correlations of class rankings across ranges:")
    range_labels = list(range_means.keys())
    for i in range(len(range_labels)):
        for j in range(i + 1, len(range_labels)):
            a, b = range_labels[i], range_labels[j]
            if a in pivot.columns and b in pivot.columns:
                r, p = spearmanr(pivot[a], pivot[b])
                log(buf, f"  {a} vs {b}: r={r:.4f}  p={p:.2e}")

    log(buf, "\nInterpretation: high Spearman r across ranges means class deviations")
    log(buf, "are consistent and not finite-size effects.")

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for idx, (label, series) in enumerate(range_means.items()):
        series_sorted = series.sort_index()
        ax.plot(series_sorted.index.astype(str), series_sorted.values,
                marker="o", markersize=4, linewidth=1, alpha=0.75,
                label=label, color=cmap(idx % 10))
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="H-L baseline")
    ax.set_xlabel("p mod 210")
    ax.set_ylabel("Mean hl_ratio")
    ax.set_title("H-L Ratio by Residue Class Across log_p Sub-ranges\n"
                 "(Consistent deviations = genuine effect beyond H-L)")
    ax.legend(fontsize=8, loc="upper right")
    plt.xticks(rotation=75, fontsize=6)
    fig.tight_layout()
    out = FIGURES_DIR / "hl_range_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(buf, f"Saved {out}")


def summarize_pysr(buf: StringIO):
    section(buf, "PySR Symbolic Regression Results")
    eq_file = RESIDUE_DIR / "pysr_equations.csv"
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
    significance_table(buf)
    singular_series_comparison(buf)
    range_ablation(df, buf)
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
