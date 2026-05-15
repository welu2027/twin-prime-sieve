"""
Publication-quality figures for the twin prime residue class paper.

  fig1_gap_deviation.png   -- lollipop: mean hl_ratio per class w/ error bands
  fig2_significance.png    -- volcano: Bonferroni significance
  fig3_ablation.png        -- bump chart: class rank stability across 1e9–1e12
  fig4_pysr_fit.png        -- scatter + residual lollipop: PySR formula fit
  fig5_los.png             -- transition heatmap + anti-persistence scatter

Run after los_comparison.py:
  python los_comparison.py
  python paper_figures.py
"""
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import COUNTS_DIR, FIGURES_DIR, RESIDUE_DIR

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 180,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})

BLUE  = "#2166ac"
RED   = "#d6604d"
TEAL  = "#4dac26"
GRAY  = "#999999"
LGRAY = "#dddddd"


def _save(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 1: Lollipop — gap deviation per class ──────────────────────────────

def fig1_gap_deviation():
    df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv").sort_values("mean_hl_ratio")
    dev = df["mean_hl_ratio"].values - 1.0
    sem = (df["std"] / np.sqrt(df["n"])).values
    labels = df["mod210"].astype(str).values
    y = np.arange(len(df))
    colors = [RED if v > 0 else BLUE for v in dev]

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    for yi, (d, s, c) in enumerate(zip(dev, sem, colors)):
        ax.plot([0, d], [yi, yi], color=c, linewidth=1.2, zorder=2)
        ax.errorbar(d, yi, xerr=2 * s, fmt="none", color=c,
                    capsize=2.5, linewidth=0.8, zorder=3)
        ax.scatter(d, yi, color=c, s=55, zorder=4, edgecolors="white", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean gap hl_ratio $-$ 1  (deviation from H-L prediction)")
    ax.set_ylabel("$p\\,\\mathrm{mod}\\,210$ residue class")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=7, label="Above H-L"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=7, label="Below H-L"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig1_gap_deviation.png")


# ── Figure 2: Volcano — Bonferroni significance ───────────────────────────────

def fig2_significance():
    df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    alpha_bonf = 0.01 / len(df)
    bonf_line  = -np.log10(alpha_bonf)

    neg_log_p = -np.log10(df["p_value"].clip(lower=1e-300))
    deviation = df["mean_hl_ratio"] - 1.0
    bonf_sig  = neg_log_p >= bonf_line

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(deviation[~bonf_sig], neg_log_p[~bonf_sig],
               color=LGRAY, edgecolors=GRAY, s=45, linewidths=0.6,
               label="Below Bonferroni threshold", zorder=2)
    ax.scatter(deviation[bonf_sig & (deviation < 0)], neg_log_p[bonf_sig & (deviation < 0)],
               color=BLUE, edgecolors="white", s=65, linewidths=0.5,
               label=f"Significant, below H-L ({(bonf_sig & (deviation<0)).sum()})", zorder=3)
    ax.scatter(deviation[bonf_sig & (deviation > 0)], neg_log_p[bonf_sig & (deviation > 0)],
               color=RED, edgecolors="white", s=65, linewidths=0.5,
               label=f"Significant, above H-L ({(bonf_sig & (deviation>0)).sum()})", zorder=3)

    for _, row in df[bonf_sig].iterrows():
        nlp = -np.log10(max(row["p_value"], 1e-300))
        ax.annotate(str(int(row["mod210"])),
                    xy=(row["mean_hl_ratio"] - 1.0, nlp),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=7, color="#333333")

    ax.axhline(bonf_line, color=RED, linewidth=1.0, linestyle="--",
               label=f"Bonferroni $\\alpha$={alpha_bonf:.2e}", zorder=1)
    ax.axvline(0, color="black", linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlabel("Mean hl_ratio $-$ 1")
    ax.set_ylabel("$-\\log_{10}(p)$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig2_significance.png")


# ── Figure 3: Bump chart — class rank stability across 1e9–1e12 ───────────────

def fig3_ablation():
    range_files = sorted(COUNTS_DIR.glob("rust_class_counts_*.csv"))
    if not range_files:
        print("No rust_class_counts CSVs. Run rust_class_counts.py --ablation first.")
        return

    combined = pd.concat([pd.read_csv(f) for f in range_files], ignore_index=True)
    combined = combined.dropna(subset=["range"])
    range_labels = sorted(combined["range"].unique())
    classes = sorted(combined["mod210"].unique())

    # Pivot to (class × range) hl_ratio, then rank within each range
    pivot = combined.pivot_table(index="mod210", columns="range", values="hl_ratio")
    pivot = pivot[range_labels].dropna()
    ranks = pivot.rank(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bump chart (rank over ranges)
    cmap = plt.get_cmap("tab20")
    x_pos = np.arange(len(range_labels))
    for i, cls in enumerate(pivot.index):
        ys = ranks.loc[cls].values
        color = cmap(i / len(pivot.index))
        axes[0].plot(x_pos, ys, color=color, linewidth=1.3, alpha=0.8, marker="o",
                     markersize=5, markeredgecolor="white", markeredgewidth=0.4)
        axes[0].annotate(str(int(cls)), xy=(x_pos[-1], ys[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=6.5, va="center", color=color)

    axes[0].set_xticks(x_pos)
    short = [l.replace("[1e", "$10^{").replace(",1e", "}$–$10^{").replace(")", "}$")
             for l in range_labels]
    axes[0].set_xticklabels(short, rotation=20, ha="right")
    axes[0].set_ylabel("Rank (hl_ratio, low = below H-L)")
    axes[0].set_xlabel("Decade range")
    axes[0].invert_yaxis()

    # Right: Spearman correlation heatmap
    n = len(range_labels)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and range_labels[i] in pivot.columns and range_labels[j] in pivot.columns:
                r, _ = spearmanr(pivot[range_labels[i]], pivot[range_labels[j]])
                corr_matrix[i, j] = r

    im = axes[1].imshow(corr_matrix, vmin=0.5, vmax=1.0,
                        cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    axes[1].set_yticklabels(short, fontsize=8)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if corr_matrix[i,j] > 0.75 else "black")
    cb = plt.colorbar(im, ax=axes[1], shrink=0.8)
    cb.set_label("Spearman $r$")
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Range")

    fig.tight_layout()
    _save(fig, "fig3_ablation.png")


# ── Figure 4: PySR fit — scatter + residual lollipop ─────────────────────────

def fig4_pysr_fit():
    stats_df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    eq_df    = pd.read_csv(RESIDUE_DIR / "residue_class_equations.csv")
    best_eq  = eq_df.sort_values("complexity").iloc[-1]["equation"]

    r_vals      = stats_df["mod210"].values
    mod70_r     = r_vals % 70
    dist_7      = np.array([min(r % 7, 7 - r % 7) for r in r_vals], dtype=float)
    dist_7_safe = np.where(dist_7 == 0, 1e-9, dist_7)
    predicted   = -0.00046384882 * r_vals + 1.0286833 + 0.0009837686 * mod70_r / dist_7_safe
    actual      = stats_df["mean_hl_ratio"].values

    r_sp, _ = spearmanr(actual, predicted)
    residuals = actual - predicted
    sort_idx  = np.argsort(actual)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter with identity line
    lo = min(actual.min(), predicted.min()) - 0.003
    hi = max(actual.max(), predicted.max()) + 0.003
    axes[0].plot([lo, hi], [lo, hi], color=GRAY, linewidth=1.2, linestyle="--",
                 label="Perfect fit", zorder=1)
    axes[0].scatter(actual, predicted, color=BLUE, s=60, zorder=3,
                    edgecolors="white", linewidths=0.6)
    for i, row in stats_df.iterrows():
        axes[0].annotate(str(int(row["mod210"])),
                         xy=(actual[i], predicted[i]),
                         xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[0].set_xlabel("Observed mean hl_ratio")
    axes[0].set_ylabel("PySR predicted hl_ratio")
    axes[0].legend()
    axes[0].text(0.05, 0.95, f"Spearman $r$ = {r_sp:.3f}",
                 transform=axes[0].transAxes, fontsize=9,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: residual lollipop
    y = np.arange(len(residuals))
    res_sorted = residuals[sort_idx]
    labels_sorted = stats_df["mod210"].values[sort_idx].astype(str)
    colors = [RED if v > 0 else BLUE for v in res_sorted]
    axes[1].axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    for yi, (res, c) in enumerate(zip(res_sorted, colors)):
        axes[1].plot([0, res], [yi, yi], color=c, linewidth=1.1, zorder=2)
        axes[1].scatter(res, yi, color=c, s=45, zorder=3,
                        edgecolors="white", linewidths=0.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels_sorted, fontsize=7)
    axes[1].set_xlabel("Residual (observed $-$ predicted)")
    axes[1].set_ylabel("$p\\,\\mathrm{mod}\\,210$ (sorted by observed hl_ratio)")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].grid(axis="y", alpha=0)

    fig.tight_layout()
    _save(fig, "fig4_pysr_fit.png")


# ── Figure 5: LO&S — transition heatmap + anti-persistence scatter ────────────

def fig5_los():
    los_file   = RESIDUE_DIR / "los_comparison.csv"
    trans_file = RESIDUE_DIR / "los_transition.csv"
    if not los_file.exists() or not trans_file.exists():
        print("LO&S files not found. Run: python los_comparison.py")
        return

    comp_df  = pd.read_csv(los_file)
    trans_df = pd.read_csv(trans_file, index_col=0)
    trans_df.columns = trans_df.columns.astype(int)
    trans_df.index   = trans_df.index.astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: transition matrix heatmap
    classes = sorted(trans_df.index)
    mat = trans_df.loc[classes, classes].values
    n   = len(classes)
    uniform = 1.0 / n
    # Center colormap around uniform rate
    vmax = max(abs(mat - uniform).max() * 2, 0.01)
    im = axes[0].imshow(mat, cmap="RdBu_r",
                        vmin=uniform - vmax/2, vmax=uniform + vmax/2,
                        aspect="auto")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels([str(c) for c in classes], rotation=60, ha="right", fontsize=7)
    axes[0].set_yticklabels([str(c) for c in classes], fontsize=7)
    axes[0].set_xlabel("Next twin prime class $p_{n+1}\\,\\mathrm{mod}\\,210$")
    axes[0].set_ylabel("Current class $p_n\\,\\mathrm{mod}\\,210$")
    cb = plt.colorbar(im, ax=axes[0], shrink=0.85)
    cb.set_label("Transition probability")

    # Highlight diagonal
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor="black", linewidth=1.2))

    # Right: diagonal rate vs mean hl_ratio scatter
    r_s, p_s = spearmanr(comp_df["diagonal_rate"], comp_df["mean_hl_ratio"])
    axes[1].axvline(uniform, color=GRAY, linewidth=1.0, linestyle=":",
                    label=f"Uniform baseline ({uniform:.3f})")
    axes[1].scatter(comp_df["diagonal_rate"], comp_df["mean_hl_ratio"],
                    color=TEAL, s=65, zorder=3, edgecolors="white", linewidths=0.6)
    for _, row in comp_df.iterrows():
        axes[1].annotate(str(int(row["mod210"])),
                         xy=(row["diagonal_rate"], row["mean_hl_ratio"]),
                         xytext=(4, 2), textcoords="offset points", fontsize=6.5)
    axes[1].set_xlabel("Self-transition rate $P(r \\to r)$")
    axes[1].set_ylabel("Mean gap hl_ratio")
    axes[1].text(0.05, 0.95, f"Spearman $r$ = {r_s:.3f}\n$p$ = {p_s:.2e}",
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "fig5_los.png")


def main():
    print("Generating paper figures...")
    fig1_gap_deviation()
    fig2_significance()
    fig3_ablation()
    fig4_pysr_fit()
    fig5_los()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
