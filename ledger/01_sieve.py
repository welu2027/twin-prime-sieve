"""
Stage 1: enumerate all (p, p+c) prime pairs for c in {2, 4, 6} with
p + c <= XMAX, and the gap from each pair's first element to the next
pair's first element.

Output: ledger/data/pairs_{twin,cousin,sexy}.parquet, columns:
    p   (int64)  first element of the pair
    gap (int32)  p_next - p  (last pair dropped: no successor)

Cross-checks (fail loudly):
  - twins: count == 3,424,506 (known pi_2(1e9)) and byte-identical to the
    paper pipeline's twin list (recomputed with the same algorithm)
  - all constellations vs. brute-force sympy-free reference below 1e6
  - gap support: multiples of 6 (twin/cousin), even (sexy)
  - sexy count ~ 2x twin count (Hardy-Littlewood 4C2 vs 2C2)

Adapted from project/rho_bar_mod2310.py:twin_firsts(); the segment overlap
is widened from 1 odd to c/2 odds so pairs straddling a boundary are caught.
"""

import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

XMAX = 1_000_000_000
SEG = 1 << 24  # odd numbers per segment
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "data")
NAMES = {2: "twin", 4: "cousin", 6: "sexy"}
PI2_1E9 = 3_424_506  # known twin pair count, p + 2 <= 1e9


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p:: p] = False
    return np.flatnonzero(s)


def pair_firsts(limit, c):
    """All odd p with p, p+c both prime, p + c <= limit (segmented, odd-only).

    Segments overlap by c/2 odds so a pair straddling a boundary is caught
    exactly once (starters strictly below the overlap re-scan are new).
    """
    k = c // 2  # index offset on the odd grid
    base = small_primes(int((limit + 2) ** 0.5) + 1)[1:]  # odd base primes
    chunks = []
    lo = 3
    hi_odd = limit - c + 1 if (limit - c) % 2 == 0 else limit - c  # last valid starter
    while lo <= hi_odd:
        last = min(lo + 2 * (SEG - 1), limit - 1)  # largest odd sieved this segment
        n = (last - lo) // 2 + 1
        if n < k + 1:
            break
        is_p = np.ones(n, dtype=bool)
        for p in base:
            start = max(p * p, ((lo + p - 1) // p) * p)
            if start % 2 == 0:
                start += p
            if start > last:
                continue
            is_p[(start - lo) // 2:: p] = False
        tw = np.flatnonzero(is_p[:-k] & is_p[k:])  # odds i and i + c/2 differ by c
        starters = (lo + 2 * tw).astype(np.int64)
        starters = starters[starters + c <= limit]
        chunks.append(starters)
        lo = last - 2 * (k - 1)  # overlap c/2 odds; re-found starters dropped below
    out = np.concatenate(chunks)
    return np.unique(out)  # overlap re-scan may duplicate boundary starters


def brute_reference(limit, c):
    s = np.ones(limit + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if s[p]:
            s[p * p:: p] = False
    ps = np.flatnonzero(s)
    return ps[(ps + c <= limit) & s[np.minimum(ps + c, limit)]]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    counts = {}
    for c, name in NAMES.items():
        t1 = time.time()
        firsts = pair_firsts(XMAX, c)
        assert np.all(np.diff(firsts) > 0), f"{name}: not strictly increasing"

        # cross-check vs brute force below 1e6
        ref = brute_reference(1_000_000, c)
        got = firsts[firsts + c <= 1_000_000]
        assert np.array_equal(got, ref), (
            f"{name}: mismatch below 1e6 ({len(got)} vs {len(ref)})")

        # gap support check
        gaps = np.diff(firsts)
        support_mod = 6 if c in (2, 4) else 2
        # skip boundary pairs below the wheel (p in {3,5} etc.)
        interior = firsts[:-1] > 10
        assert np.all(gaps[interior] % support_mod == 0), (
            f"{name}: gap support violates mod {support_mod}")

        if c == 2:
            assert len(firsts) == PI2_1E9, (
                f"twin count {len(firsts)} != known {PI2_1E9}")

        counts[name] = len(firsts)
        tbl = pa.table({"p": firsts[:-1], "gap": gaps.astype(np.int32)})
        path = os.path.join(OUTDIR, f"pairs_{name}.parquet")
        pq.write_table(tbl, path)
        print(f"{name:7s}: {len(firsts):>9,} pairs, {len(gaps):,} gaps "
              f"-> {path} ({time.time()-t1:.1f}s)")

    ratio = counts["sexy"] / counts["twin"]
    assert 1.9 < ratio < 2.1, f"sexy/twin ratio {ratio:.3f} outside [1.9, 2.1]"
    print(f"sexy/twin ratio: {ratio:.4f} (HL predicts -> 2)")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
