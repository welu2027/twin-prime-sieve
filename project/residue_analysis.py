"""
Stage 1 of the closed-form H-L correction search.

Instead of treating mod210 as a continuous number (which PySR can't interpret
as a category), we:
  1. Compute mean hl_ratio per mod210 class
  2. Engineer mathematical features OF each class (distance to multiples of
     small primes, number-theoretic properties)
  3. Run PySR on the 48 class-level data points to find what predicts the
     correction factor

If PySR finds a clean formula here, that IS a closed-form H-L correction.

Output:
  data/residue_class_stats.csv   — per-class statistics
  data/residue_class_pysr.txt    — PySR results on class-level features
"""
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from scipy import stats

from utils import DATA_DIR, RESIDUE_DIR

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]  # primes dividing 2310


def class_features(r: int) -> dict:
    """Compute number-theoretic features of a residue class r mod 210."""
    feats = {"mod210": r}
    # distance to nearest multiple of each small prime
    for p in [7, 11, 13, 17, 19, 23]:
        dist = min(r % p, p - r % p)
        feats[f"dist_{p}"] = dist
    # sum of distances (how "isolated" this class is from multiples of small primes)
    feats["isolation"] = sum(min(r % p, p - r % p) for p in [7, 11, 13, 17, 19, 23])
    # r mod 30, r mod 70, r mod 42 (sub-wheel structure)
    feats["mod30_r"]  = r % 30
    feats["mod42_r"]  = r % 42
    feats["mod70_r"]  = r % 70
    # digit sum of r (weak proxy for multiplicative structure)
    feats["digit_sum"] = sum(int(d) for d in str(r))
    return feats


def main():
    print("Loading features parquet...")
    df = pd.read_parquet(DATA_DIR / "twin_primes_features.parquet")
    df = df[(df["hl_ratio"] > 0) & (df["hl_ratio"] < 10) & (df["log_p"] > 8)]

    # per-class statistics
    grp = df.groupby("mod210")["hl_ratio"].agg(
        mean="mean", std="std", count="count", median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()

    # t-test: is each class's mean significantly different from 1.0?
    results = []
    for r in grp["mod210"]:
        vals = df[df["mod210"] == r]["hl_ratio"].values
        if len(vals) < 30:
            continue
        t, p = stats.ttest_1samp(vals, 1.0)
        row = {"mod210": r, "mean_hl_ratio": vals.mean(), "std": vals.std(),
               "n": len(vals), "t_stat": t, "p_value": p,
               "significant": p < 0.01}
        row.update(class_features(r))
        results.append(row)

    class_df = pd.DataFrame(results)
    class_df = class_df.sort_values("mean_hl_ratio")

    out_csv = RESIDUE_DIR / "residue_class_stats.csv"
    class_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    sig = class_df[class_df["significant"]]
    print(f"\n{len(sig)}/{len(class_df)} residue classes significantly deviate from H-L (p<0.01):")
    print(sig[["mod210", "mean_hl_ratio", "t_stat", "p_value", "n"]].to_string(index=False))

    print("\n--- Running PySR on class-level features ---")
    print("(48 data points — one per residue class)")
    print("Target: mean hl_ratio per class\n")

    feat_cols = ["mod210", "dist_7", "dist_11", "dist_13", "dist_17",
                 "dist_19", "dist_23", "isolation", "mod30_r", "mod42_r", "mod70_r"]
    X = class_df[feat_cols].values.astype(np.float32)
    y = class_df["mean_hl_ratio"].values.astype(np.float32)

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "square"],
        populations=20,
        population_size=50,
        maxsize=12,
        parsimony=0.002,
        random_state=42,
        verbosity=1,
        temp_equation_file=True,
    )
    model.fit(X, y, variable_names=feat_cols)

    out_txt = RESIDUE_DIR / "residue_class_pysr.txt"
    lines = [
        "PySR on class-level residue features",
        "="*60,
        "Target: mean hl_ratio per mod210 class (48 points)",
        "Features: number-theoretic properties of each class",
        "",
        "Top equations:",
        model.equations_[["equation", "loss", "complexity"]].to_string(index=False),
        f"\nBest: {model.sympy()}",
        "",
        "Interpretation: if the best equation involves dist_7, dist_11, etc.,",
        "it means gaps are larger/smaller when p is close to multiples of those primes.",
        "That would be a genuine improvement over H-L.",
    ]
    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text)
    print(f"\nSaved {out_txt}")

    model.equations_.to_csv(RESIDUE_DIR / "residue_class_equations.csv", index=False)


if __name__ == "__main__":
    main()
