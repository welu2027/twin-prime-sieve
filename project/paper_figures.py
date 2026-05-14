"""
Publication-quality figures for the twin prime residue class paper.

Generates 4 figures saved to data/figures/:
  fig1_gap_deviation.png       -- mean hl_ratio per mod210 class w/ error bars
  fig2_significance.png        -- Bonferroni volcano plot
  fig3_ablation_10e12.png      -- count-based hl_ratio per class across 1e9–1e12 ranges
  fig4_pysr_fit.png            -- PySR best formula vs actual class hl_ratio

Usage:
  python paper_figures.py
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import COUNTS_DIR, FIGURES_DIR, RESIDUE_DIR

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2166ac"
RED = "#d6604d"
GRAY = "#888888"


# ── Figure 1: Gap-based hl_ratio per class with error bars ────────────────────

def fig1_gap_deviation():
    df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    df = df.sort_values("mean_hl_ratio")
    x = np.arange(len(df))
    colors = [RED if v > 1.0 else BLUE for v in df["mean_hl_ratio"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, df["mean_hl_ratio"] - 1.0, color=colors, width=0.7, zorder=2)
    ax.errorbar(x, df["mean_hl_ratio"] - 1.0,
                yerr=df["std"] / np.sqrt(df["n"]),
                fmt="none", color="black", capsize=2, linewidth=0.8, zorder=3)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mod210"].astype(str), rotation=60, ha="right")
    ax.set_xlabel("$p \\,\\mathrm{mod}\\, 210$ (residue class)")
    ax.set_ylabel("Mean hl_ratio $-$ 1  (deviation from H-L)")
    ax.set_title("Gap-Spacing Deviation from Hardy-Littlewood by mod 210 Class\n"
                 "(error bars = SEM; blue = below H-L, red = above H-L)")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_gap_deviation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Bonferroni volcano plot ─────────────────────────────────────────

def fig2_significance():
    df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    alpha_bonf = 0.01 / len(df)
    bonf_line = -np.log10(alpha_bonf)
    nominal_line = -np.log10(0.01)

    neg_log_p = -np.log10(df["p_value"].clip(lower=1e-300))
    deviation = df["mean_hl_ratio"] - 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    bonf_sig = neg_log_p >= bonf_line
    ax.scatter(deviation[~bonf_sig], neg_log_p[~bonf_sig],
               color=GRAY, s=40, label="Not significant (Bonferroni)", zorder=2)
    ax.scatter(deviation[bonf_sig], neg_log_p[bonf_sig],
               color=RED, s=60, label=f"Bonferroni significant ({bonf_sig.sum()}/{len(df)})", zorder=3)

    for _, row in df[bonf_sig].iterrows():
        ax.annotate(str(int(row["mod210"])),
                    xy=(row["mean_hl_ratio"] - 1.0, -np.log10(max(row["p_value"], 1e-300))),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.axhline(bonf_line, color="red", linewidth=1.2, linestyle="--",
               label=f"Bonferroni threshold (α={alpha_bonf:.4g})")
    ax.axhline(nominal_line, color=GRAY, linewidth=0.8, linestyle=":",
               label="Nominal p<0.01")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Mean hl_ratio $-$ 1  (deviation from H-L)")
    ax.set_ylabel("$-\\log_{10}$(p-value)")
    ax.set_title("Significance of Residue Class Deviations\n(t-test vs H-L prediction of 1.0)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_significance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Count-based hl_ratio across 1e9–1e12 (Rust ablation) ────────────

def fig3_ablation():
    range_files = sorted(COUNTS_DIR.glob("rust_class_counts_*.csv"))
    if not range_files:
        print("No rust_class_counts CSVs found. Run: python rust_class_counts.py --ablation ...")
        return

    all_dfs = []
    for f in range_files:
        df = pd.read_csv(f)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.dropna(subset=["range"])

    range_labels = sorted(combined["range"].unique())
    classes = sorted(combined["mod210"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: line plot of hl_ratio per class per range
    cmap = plt.get_cmap("viridis")
    for idx, label in enumerate(range_labels):
        sub = combined[combined["range"] == label].set_index("mod210")
        ys = [sub.loc[c, "hl_ratio"] if c in sub.index else np.nan for c in classes]
        color = cmap(idx / max(len(range_labels) - 1, 1))
        axes[0].plot(range(len(classes)), ys, marker="o", markersize=3,
                     linewidth=1.2, label=label, color=color)
    axes[0].axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="H-L = 1.0")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels([str(c) for c in classes], rotation=60, ha="right")
    axes[0].set_xlabel("$p \\,\\mathrm{mod}\\, 210$ (residue class)")
    axes[0].set_ylabel("Count-based hl_ratio")
    axes[0].set_title("Count-Based H-L Ratio per Class Across Decades")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: Spearman r heatmap between ranges
    pivot = combined.pivot_table(index="mod210", columns="range", values="hl_ratio")
    pivot = pivot.dropna()
    n = len(range_labels)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and range_labels[i] in pivot.columns and range_labels[j] in pivot.columns:
                r, _ = spearmanr(pivot[range_labels[i]], pivot[range_labels[j]])
                corr_matrix[i, j] = r

    im = axes[1].imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    short_labels = [l.replace("[1e", "10^").replace(",1e", "–10^").replace(")", "") for l in range_labels]
    axes[1].set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticklabels(short_labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=axes[1], label="Spearman r")
    axes[1].set_title("Spearman Rank Correlation of Class hl_ratio\nAcross Decade Ranges")

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_ablation_10e12.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: PySR formula fit vs actual class hl_ratio ───────────────────────

def fig4_pysr_fit():
    stats_df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    eq_df = pd.read_csv(RESIDUE_DIR / "residue_class_equations.csv")

    # Best equation: complexity 11
    best_row = eq_df.sort_values("complexity").iloc[-1]
    best_eq = best_row["equation"]

    # Compute predicted values using the best lambda
    import ast
    import re
    # Use sympy_format to evaluate
    r_vals = stats_df["mod210"].values
    mod70_r = r_vals % 70
    dist_7 = np.array([min(r % 7, 7 - r % 7) for r in r_vals], dtype=float)
    dist_7_safe = np.where(dist_7 == 0, 1e-9, dist_7)

    # Best formula: -0.00046384882*mod210 + 1.0286833 + 0.0009837686*mod70_r/dist_7
    predicted = -0.00046384882 * r_vals + 1.0286833 + 0.0009837686 * mod70_r / dist_7_safe
    actual = stats_df["mean_hl_ratio"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter predicted vs actual
    axes[0].scatter(actual, predicted, color=BLUE, s=50, zorder=3)
    lim = [min(actual.min(), predicted.min()) - 0.005,
           max(actual.max(), predicted.max()) + 0.005]
    axes[0].plot(lim, lim, "r--", linewidth=1.2, label="Perfect fit")
    r, p = spearmanr(actual, predicted)
    axes[0].set_xlabel("Actual mean hl_ratio")
    axes[0].set_ylabel("PySR predicted hl_ratio")
    axes[0].set_title(f"PySR Best Formula vs Actual\n(Spearman r={r:.3f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: residuals by class
    residuals = actual - predicted
    x = np.argsort(actual)
    axes[1].bar(range(len(residuals)), residuals[x], color=[
        RED if v > 0 else BLUE for v in residuals[x]], width=0.7)
    axes[1].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(range(len(stats_df)))
    axes[1].set_xticklabels(stats_df["mod210"].values[x].astype(str), rotation=60, ha="right")
    axes[1].set_xlabel("mod210 class (sorted by actual hl_ratio)")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_title("Residuals After PySR Correction\n"
                      f"Formula: {best_eq[:60]}{'...' if len(best_eq)>60 else ''}")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_pysr_fit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("Generating paper figures...")
    fig1_gap_deviation()
    fig2_significance()
    fig3_ablation()
    fig4_pysr_fit()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
