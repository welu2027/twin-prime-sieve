"""
Use the Rust SSoZ --per-class mode to get per-mod210 twin prime counts at large scale.
Supports ranges up to 10^12+ (much faster than Python sieve).

The Rust binary counts twin primes in each mod210 residue class in one sieve pass.
We compare observed counts to the H-L expected count (2C₂/(ln p)² integrated over range).

Usage:
  # 10^12 full run (takes ~10-30 min on a laptop):
  python rust_class_counts.py --end 1e12

  # Range run:
  python rust_class_counts.py --start 1e9 --end 1e12

  # Ablation — multiple ranges to check consistency:
  python rust_class_counts.py --ablation 1e6 1e7 1e8 1e9 1e10 1e11 1e12

Must build Rust binary first:
  cd ../ && cargo build --release && cp target/release/twinprimes_ssoz project/twinprimes_ssoz
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

from utils import BINARY, COUNTS_DIR

C2 = 0.6601618158468696


def expected_hl_count(a: float, b: float) -> float:
    """Integrate 2C₂/(ln t)² from a to b for H-L twin prime count estimate."""
    a = max(a, 5.0)
    result, _ = integrate.quad(lambda t: 2 * C2 / (np.log(t) ** 2), a, b)
    return result


def run_per_class(start: int, end: int, verbose: bool = True) -> pd.DataFrame:
    """Run Rust binary with --per-class for [start, end], return DataFrame."""
    if not BINARY.exists():
        raise FileNotFoundError(
            f"Rust binary not found at {BINARY}.\n"
            "Build with: cd twin-prime-sieve && cargo build --release && "
            "cp target/release/twinprimes_ssoz project/twinprimes_ssoz"
        )
    cmd = [str(BINARY), str(start), str(end), "--per-class"]
    if verbose:
        print(f"Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Rust binary failed (exit {result.returncode})")

    if verbose:
        # Print non-CSV lines (progress/timing) to stdout
        for line in result.stdout.splitlines():
            if not line:
                continue
            if line != "mod210,count" and not (line[0].isdigit() and "," in line):
                print(line)

    # Parse the CSV section
    csv_lines = []
    in_csv = False
    for line in result.stdout.splitlines():
        if line == "mod210,count":
            in_csv = True
        if in_csv:
            csv_lines.append(line)

    if not csv_lines:
        raise RuntimeError("No per-class CSV output found. "
                           "Is the range > 77M? (needs modpg >= 210)")

    df = pd.read_csv(io.StringIO("\n".join(csv_lines)))
    return df


def compute_hl_ratios(counts_df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Add expected_hl and hl_ratio columns. Assumes uniform per-class H-L baseline."""
    df = counts_df.copy()
    total_expected = expected_hl_count(float(start), float(end))
    n_classes = len(df)
    df["expected_hl"] = total_expected / n_classes
    df["hl_ratio"] = df["count"] / df["expected_hl"]
    return df


def print_summary(df: pd.DataFrame, start: int, end: int):
    total_obs = df["count"].sum()
    total_exp = df["expected_hl"].sum()
    print(f"\nRange: [{start:,}, {end:,}]  (log₁₀ {np.log10(start):.1f}–{np.log10(end):.1f})")
    print(f"Total observed  : {total_obs:,}")
    print(f"Total H-L expected: {total_exp:,.0f}  (ratio={total_obs/total_exp:.4f})")
    print(f"\nPer-class hl_ratio (H-L predicts 1.0 for all):")
    print(df.sort_values("hl_ratio")[["mod210", "count", "expected_hl", "hl_ratio"]].to_string(index=False))
    below = df[df["hl_ratio"] < 0.99].sort_values("hl_ratio")
    above = df[df["hl_ratio"] > 1.01].sort_values("hl_ratio", ascending=False)
    print(f"\nClasses below H-L by >1%: {below['mod210'].tolist()}")
    print(f"Classes above H-L by >1%: {above['mod210'].tolist()}")


def run_ablation(thresholds: list[float]):
    """Run per-class sieve on each consecutive range, print consistency check."""
    ranges = list(zip(thresholds[:-1], thresholds[1:]))
    all_results = {}

    for lo, hi in ranges:
        start, end = int(lo), int(hi)
        label = f"[1e{np.log10(lo):.0f},1e{np.log10(hi):.0f})"
        print(f"\n{'='*60}")
        print(f"Range {label}")
        print("="*60)
        try:
            raw = run_per_class(start, end)
            df = compute_hl_ratios(raw, start, end)
            print_summary(df, start, end)
            all_results[label] = df.set_index("mod210")["hl_ratio"]
            df["range"] = label
            out = COUNTS_DIR / f"rust_class_counts_{start}_{end}.csv"
            df.to_csv(out, index=False)
            print(f"Saved {out}")
        except Exception as e:
            print(f"Skipped {label}: {e}")

    if len(all_results) >= 2:
        print("\n" + "="*60)
        print("Spearman rank correlations of class hl_ratio across ranges:")
        print("(High r = consistent deviations = real effect, not finite-size)")
        labels = list(all_results.keys())
        pivot = pd.DataFrame(all_results).dropna()
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                if a in pivot.columns and b in pivot.columns:
                    r, p = stats.spearmanr(pivot[a], pivot[b])
                    print(f"  {a} vs {b}: r={r:.4f}  p={p:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Per-mod210 twin prime counts via Rust SSoZ.")
    parser.add_argument("--start", type=float, default=1e9,
                        help="Range start (default: 1e9)")
    parser.add_argument("--end", type=float, default=1e12,
                        help="Range end (default: 1e12)")
    parser.add_argument("--ablation", type=float, nargs="+",
                        metavar="N",
                        help="Run on consecutive ranges: e.g. --ablation 1e6 1e7 1e8 1e9")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: data/rust_class_counts_START_END.csv)")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.ablation)
        return

    start, end = int(args.start), int(args.end)
    raw = run_per_class(start, end)
    df = compute_hl_ratios(raw, start, end)
    print_summary(df, start, end)

    out = args.out or str(COUNTS_DIR / f"rust_class_counts_{start}_{end}.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
