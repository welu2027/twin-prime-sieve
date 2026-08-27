"""
Stage 2: full hazard-model prediction table for every ledger cell.

For each (constellation, modulus, class, window) computes, from the full
predicted gap distribution (mixture over density-weighted log-scale nodes,
weight e^u/u^2 — same scheme as the paper's preregistration):

    mean_norm_gap                — paper's rho_bar_r (class mean / grand mean)
    var/skew/kurt_norm_gap       — central moments of rho = g*S_c/u^2
    p_tail_{2,3,5}x              — P(rho > t * cell mean rho)
    p_gap_<v>                    — P(gap = v), v in the constellation's set
    hazard (one row per grid g)  — P(gap = g)/P(gap >= g)
    autocorr_lag1                — 0 by construction (hazard model = renewal)

plus the wheel-only null (C == 1) for every statistic.

Windows: decades 1e5-1e6 .. 1e8-1e9 (5 nodes each) and cumulative le_1e9
(UMIN = 11.5, 13 nodes — identical to predict_rho_bar_mod2310.py).

Output: ledger/data/predicted.parquet (long format).
Deterministic: no RNG anywhere.
"""

import math
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import hazard as hz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "predicted.parquet")

UMIN = 11.5
MODULI = [210, 2310]
TAILS = [2, 3, 5]

WINDOWS = [
    ("1e5-1e6", math.log(1e5), math.log(1e6), 5),
    ("1e6-1e7", math.log(1e6), math.log(1e7), 5),
    ("1e7-1e8", math.log(1e7), math.log(1e8), 5),
    ("1e8-1e9", math.log(1e8), math.log(1e9), 5),
    ("le_1e9", UMIN, math.log(1e9), 13),
]

HAZARD_MULTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 24, 29, 35, 42, 50, 60]


def gap_sets(c):
    hazard_grid = [6 * m for m in HAZARD_MULTS]
    small = [6, 12, 18, 30]
    if c == 6:
        hazard_grid = [2, 4] + hazard_grid
        small = [2, 4] + small
    return np.array(hazard_grid), small


def window_nodes(lo, hi, n):
    us = np.linspace(lo, hi, n)
    wts = np.exp(us) / us**2
    return us, wts / wts.sum()


def cell_statistics(r, c, mod, us, wts, wheel_only):
    """All predicted statistics for one class-window cell."""
    S = hz.S_const(c)
    hazard_grid, small = gap_sets(c)

    # per-node distributions (supports differ in length; collect then mix)
    moments = np.zeros(5)          # E[rho^k], k = 0..4
    pg_small = np.zeros(len(small))
    pg_hz = np.zeros(len(hazard_grid))
    surv_hz = np.zeros(len(hazard_grid))
    mean_gap = 0.0
    dists = []
    for u, w in zip(us, wts):
        gs, P = hz.class_gap_distribution(r, u, c, mod, wheel_only)
        rho = gs * (S / u**2)
        dists.append((gs, P, rho, w))
        mean_gap += w * float((gs * P).sum())
        for k in range(5):
            moments[k] += w * float((P * rho**k).sum())
        for i, v in enumerate(small):
            j = np.searchsorted(gs, v)
            if j < len(gs) and gs[j] == v:
                pg_small[i] += w * P[j]
        cum = np.concatenate((np.cumsum(P[::-1])[::-1], [0.0]))  # survivor incl. g
        for i, v in enumerate(hazard_grid):
            j = np.searchsorted(gs, v)
            if j < len(gs):
                surv_hz[i] += w * cum[j]
                if gs[j] == v:
                    pg_hz[i] += w * P[j]

    mu = moments[1]
    var = moments[2] - mu**2
    sd = math.sqrt(max(var, 0.0))
    m3 = moments[3] - 3 * mu * moments[2] + 2 * mu**3
    m4 = moments[4] - 4 * mu * moments[3] + 6 * mu**2 * moments[2] - 3 * mu**4
    skew = m3 / sd**3 if sd > 0 else float("nan")
    kurt = m4 / sd**4 - 3.0 if sd > 0 else float("nan")

    tails = []
    for t in TAILS:
        acc = 0.0
        for gs, P, rho, w in dists:
            acc += w * float(P[rho > t * mu].sum())
        tails.append(acc)

    with np.errstate(divide="ignore", invalid="ignore"):
        hz_curve = np.where(surv_hz > 0, pg_hz / surv_hz, np.nan)

    stats = {"mean_gap_raw": mean_gap, "var_norm_gap": var,
             "skew_norm_gap": skew, "kurt_norm_gap": kurt,
             "autocorr_lag1": 0.0}
    for t, v in zip(TAILS, tails):
        stats[f"p_tail_{t}x"] = v
    for v, pv in zip(small, pg_small):
        stats[f"p_gap_{v}"] = pv
    return stats, dict(zip(hazard_grid.tolist(), hz_curve.tolist()))


def main():
    t0 = time.time()
    rows = {"constellation": [], "constellation_pattern": [], "modulus": [],
            "class": [], "window": [], "statistic": [], "g": [],
            "predicted": [], "predicted_wheel_only": []}

    def emit(name, c, mod, r, win, stat, g, full, wheel):
        rows["constellation"].append(name)
        rows["constellation_pattern"].append(f"(0,{c})")
        rows["modulus"].append(mod)
        rows["class"].append(r)
        rows["window"].append(win)
        rows["statistic"].append(stat)
        rows["g"].append(g)
        rows["predicted"].append(full)
        rows["predicted_wheel_only"].append(wheel)

    for name, c in hz.CONSTELLATIONS.items():
        for mod in MODULI:
            classes = hz.valid_classes(mod, c)
            for win, lo, hi, n in WINDOWS:
                us, wts = window_nodes(lo, hi, n)
                per_class = {}
                for r in classes:
                    sf, hzf = cell_statistics(r, c, mod, us, wts, False)
                    sw, hzw = cell_statistics(r, c, mod, us, wts, True)
                    per_class[r] = (sf, hzf, sw, hzw)
                grand_f = np.mean([v[0]["mean_gap_raw"] for v in per_class.values()])
                grand_w = np.mean([v[2]["mean_gap_raw"] for v in per_class.values()])
                for r, (sf, hzf, sw, hzw) in per_class.items():
                    emit(name, c, mod, r, win, "mean_norm_gap", None,
                         sf["mean_gap_raw"] / grand_f, sw["mean_gap_raw"] / grand_w)
                    for stat in sf:
                        if stat == "mean_gap_raw":
                            continue
                        emit(name, c, mod, r, win, stat, None, sf[stat], sw[stat])
                    for g in hzf:
                        emit(name, c, mod, r, win, "hazard", g, hzf[g], hzw[g])
                print(f"{name} mod {mod} {win}: {len(classes)} classes "
                      f"({time.time()-t0:.0f}s)", flush=True)

    tbl = pa.table({
        "constellation": pa.array(rows["constellation"], pa.string()),
        "constellation_pattern": pa.array(rows["constellation_pattern"], pa.string()),
        "modulus": pa.array(rows["modulus"], pa.int32()),
        "class": pa.array(rows["class"], pa.int32()),
        "window": pa.array(rows["window"], pa.string()),
        "statistic": pa.array(rows["statistic"], pa.string()),
        "g": pa.array(rows["g"], pa.int32()),
        "predicted": pa.array(rows["predicted"], pa.float64()),
        "predicted_wheel_only": pa.array(rows["predicted_wheel_only"], pa.float64()),
    })
    pq.write_table(tbl, OUT)
    print(f"wrote {OUT}: {tbl.num_rows:,} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
