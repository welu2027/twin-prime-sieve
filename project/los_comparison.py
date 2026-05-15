"""
Lemke Oliver & Soundararajan (LO&S) comparison for twin prime residue classes.

LO&S (2016) showed consecutive primes avoid repeating the same residue class.
We test whether this anti-persistence explains the gap-spacing deviations we
observe per mod-210 class.

Method:
  1. Build 15x15 transition matrix of consecutive twin prime residue classes
  2. Diagonal rate = P(next twin prime in same class) — low means anti-persistent
  3. Spearman-correlate diagonal rate with mean hl_ratio per class
  4. If correlated: LO&S explains our gap deviations
     If not: our finding is independent of LO&S (stronger novelty)

Output:
  data/residue/los_comparison.csv   -- per-class diagonal rate + hl_ratio
  data/residue/los_transition.csv   -- full 15x15 transition matrix (normalized)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from utils import DATA_DIR, RESIDUE_DIR


def build_transition_matrix(df: pd.DataFrame):
    """Build normalized 15x15 transition matrix of consecutive twin prime classes."""
    df = df.sort_values("p").reset_index(drop=True)
    classes = sorted(df["mod210"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    counts = np.zeros((n, n), dtype=float)
    mod_vals = df["mod210"].values
    for k in range(len(mod_vals) - 1):
        i = class_to_idx.get(mod_vals[k])
        j = class_to_idx.get(mod_vals[k + 1])
        if i is not None and j is not None:
            counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = counts / row_sums
    return transition, classes


def main():
    print("Loading twin primes features...")
    df = pd.read_parquet(DATA_DIR / "twin_primes_features.parquet")
    df = df[df["mod210"].notna()].copy()
    df["mod210"] = df["mod210"].astype(int)

    print(f"  {len(df):,} twin primes, {df['mod210'].nunique()} residue classes")

    transition, classes = build_transition_matrix(df)
    n = len(classes)

    # Save full transition matrix
    trans_df = pd.DataFrame(transition, index=classes, columns=classes)
    trans_df.index.name = "from_class"
    trans_out = RESIDUE_DIR / "los_transition.csv"
    trans_df.to_csv(trans_out)
    print(f"Saved {trans_out}")

    # Diagonal rates (anti-persistence: low diagonal = avoids repeating class)
    diagonal_rates = {classes[i]: transition[i, i] for i in range(n)}

    # Uniform expectation: if no bias, each class equally likely → 1/n
    uniform_rate = 1.0 / n
    print(f"\nUniform baseline diagonal rate: {uniform_rate:.4f} (1/{n})")

    # Load per-class hl_ratio
    stats_df = pd.read_csv(RESIDUE_DIR / "residue_class_stats.csv")
    stats_df = stats_df[stats_df["mod210"].isin(classes)].copy()
    stats_df["diagonal_rate"] = stats_df["mod210"].map(diagonal_rates)
    stats_df["anti_persistence"] = uniform_rate - stats_df["diagonal_rate"]

    out = RESIDUE_DIR / "los_comparison.csv"
    stats_df[["mod210", "mean_hl_ratio", "diagonal_rate", "anti_persistence"]].to_csv(out, index=False)
    print(f"Saved {out}")

    # Correlations
    valid = stats_df.dropna(subset=["diagonal_rate", "mean_hl_ratio"])
    r_s, p_s = spearmanr(valid["diagonal_rate"], valid["mean_hl_ratio"])
    r_p, p_p = pearsonr(valid["diagonal_rate"], valid["mean_hl_ratio"])

    print(f"\n{'='*55}")
    print("LO&S Anti-Persistence vs Gap Deviation (hl_ratio)")
    print("="*55)
    print(f"Spearman r = {r_s:.4f}   p = {p_s:.3e}")
    print(f"Pearson  r = {r_p:.4f}   p = {p_p:.3e}")
    print()

    if abs(r_s) > 0.5 and p_s < 0.05:
        print("Interpretation: CORRELATED — LO&S anti-persistence partially explains")
        print("the gap deviations. Cite LO&S as a contributing mechanism.")
    elif abs(r_s) < 0.3 or p_s > 0.05:
        print("Interpretation: INDEPENDENT — gap deviations are NOT explained by")
        print("LO&S anti-persistence. This strengthens novelty of the finding.")
    else:
        print("Interpretation: WEAK/AMBIGUOUS — partial overlap with LO&S.")

    print(f"\nDiagonal rates per class (uniform baseline = {uniform_rate:.4f}):")
    display = valid[["mod210", "diagonal_rate", "anti_persistence", "mean_hl_ratio"]].sort_values("diagonal_rate")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
