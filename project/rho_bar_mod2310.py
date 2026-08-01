"""
Empirical rho_bar_r for the 135 admissible twin-prime classes mod 2310.

Segmented numpy sieve to XMAX; for each consecutive pair of twin primes
(p_i, p_i+2), (p_{i+1}, p_{i+1}+2) records gap g_i = p_{i+1} - p_i binned by
r = p_i mod 2310. Normalization matches the mod-210 analysis exactly:
grand mean = UNWEIGHTED mean of the 135 class mean gaps,
rho_bar_r = class mean / grand mean, SE = class std / sqrt(n) / grand mean.

Also emits the same table truncated at p_i < 1e7 and 1e8 for the 1/log x
decay check (deviation spread should shrink ~1/log x as x grows).

Usage:  python rho_bar_mod2310.py
Output: data/residue/rho_bar_mod2310.csv          (xmax = 1e9 table)
        data/residue/rho_bar_mod2310_cutoffs.csv  (all three cutoffs)

Cross-check: per-class counts should match the Rust sieve
(./target/release/twinprimes_ssoz 1_000_000_000 --per-class-mod=2310)
up to at most 1 (the final twin of the range has no successor gap).
"""

import csv
import math
import os
import sys
import time

import numpy as np

XMAX = 1_000_000_000
CUTOFFS = [10_000_000, 100_000_000, XMAX]
MOD = 2310
SEG = 1 << 24  # odd numbers per segment (~33.5M integers)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "residue")


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p:: p] = False
    return np.flatnonzero(s)


def twin_firsts(limit):
    """All p <= limit with p, p+2 both prime (odd-only segmented sieve).

    limit must be even; odds 3 .. limit+1 are sieved so a starter at
    limit-1 still has its partner covered. Segments overlap by one odd so
    a twin pair straddling a boundary is caught by the next segment.
    """
    base = small_primes(int((limit + 2) ** 0.5) + 1)[1:]  # odd base primes
    chunks = []
    lo = 3
    while lo <= limit - 1:
        last = min(lo + 2 * (SEG - 1), limit + 1)  # largest odd in segment
        n = (last - lo) // 2 + 1
        if n < 2:
            break
        is_p = np.ones(n, dtype=bool)
        for p in base:
            start = max(p * p, ((lo + p - 1) // p) * p)
            if start % 2 == 0:
                start += p
            if start > last:
                continue
            is_p[(start - lo) // 2:: p] = False
        tw = np.flatnonzero(is_p[:-1] & is_p[1:])  # consecutive odds differ by 2
        chunks.append((lo + 2 * tw).astype(np.int64))
        lo = last  # overlap: `last` is re-checked as a starter next segment
    return np.concatenate(chunks)


def class_table(starters, gaps, classes, xcut):
    mask = starters < xcut
    s, g = starters[mask] % MOD, gaps[mask]
    means, stds, ns = {}, {}, {}
    for r in classes:
        sel = g[s == r]
        ns[r] = len(sel)
        means[r] = sel.mean()
        stds[r] = sel.std(ddof=1)
    grand = sum(means.values()) / len(classes)
    rows = []
    for r in classes:
        rho = means[r] / grand
        se = stds[r] / math.sqrt(ns[r]) / grand
        rows.append((r, ns[r], means[r], rho, se))
    return rows, grand


def main():
    t0 = time.time()
    assert XMAX % 2 == 0
    classes = [r for r in range(MOD)
               if math.gcd(r, MOD) == 1 and math.gcd(r + 2, MOD) == 1]
    assert len(classes) == 135

    twins = twin_firsts(XMAX)
    print(f"sieved {len(twins)} twin firsts <= {XMAX:.0e} in {time.time()-t0:.1f}s")
    assert np.all(np.diff(twins) > 0)

    starters, gaps = twins[:-1], np.diff(twins)

    os.makedirs(OUTDIR, exist_ok=True)
    cutoff_path = os.path.join(OUTDIR, "rho_bar_mod2310_cutoffs.csv")
    main_path = os.path.join(OUTDIR, "rho_bar_mod2310.csv")
    with open(cutoff_path, "w", newline="") as fc:
        wc = csv.writer(fc)
        wc.writerow(["mod2310", "xmax", "n_gaps", "mean_gap", "rho_bar", "se_rho_bar"])
        for xcut in CUTOFFS:
            rows, grand = class_table(starters, gaps, classes, xcut)
            spread = max(r[3] for r in rows) - min(r[3] for r in rows)
            print(f"xmax={xcut:.0e}: grand mean gap {grand:.3f}, "
                  f"rho_bar spread {spread:.4f}, n per class ~{rows[0][1]}")
            for r, n, mg, rho, se in rows:
                wc.writerow([r, f"{xcut:.0e}", n, f"{mg:.4f}", f"{rho:.6f}", f"{se:.6f}"])
            if xcut == XMAX:
                with open(main_path, "w", newline="") as fm:
                    wm = csv.writer(fm)
                    wm.writerow(["mod2310", "n_gaps", "mean_gap", "rho_bar", "se_rho_bar"])
                    for r, n, mg, rho, se in rows:
                        wm.writerow([r, n, f"{mg:.4f}", f"{rho:.6f}", f"{se:.6f}"])
    print(f"wrote {main_path}\nwrote {cutoff_path}")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
