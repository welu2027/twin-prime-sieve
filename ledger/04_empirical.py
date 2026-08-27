"""
Stage 4: empirical statistics + bootstrap standard errors for every cell.

Refuses to run until ledger/prereg/predictions_20260827.csv is committed to
git and clean (preregistration gate).  `--published-only` bypasses the gate
but computes ONLY the twin mean_norm_gap le_1e9 cells (already published in
project/data/residue/) and runs the regression test against the paper's
numbers — used to validate the machinery without touching new cells.

Statistics mirror 02_predict.py exactly (same definitions, same grids).
Per-gap normalization: rho_i = gap_i * S_c / log(p_i)^2.
mean_norm_gap keeps the paper's convention: class mean raw gap / unweighted
grand mean of class means within (constellation, modulus, window).

SEs: Poisson bootstrap (weights ~ Poisson(1), B = 1000, seeded per cell) —
statistically equivalent to the classical multinomial bootstrap at these
sample sizes and vastly faster: every statistic is a function of per-
observation feature sums, so each resample is one weighted sum (BLAS matmul).
The tail thresholds t*mean use the cell mean (mean uncertainty ~0.2% is
negligible against tail SE).  autocorr_lag1 uses pair bootstrap.
Deterministic: seed = (20260827, constellation, modulus, window, class).

Output: ledger/data/empirical.parquet
"""

import math
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hazard as hz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREREG = os.path.join("ledger", "prereg", "predictions_20260827.csv")
OUT = os.path.join(HERE, "data", "empirical.parquet")

SEED = 20260827
B = 1000
TAILS = [2, 3, 5]
MODULI = [210, 2310]
WINDOWS = [("1e5-1e6", 1e5, 1e6), ("1e6-1e7", 1e6, 1e7),
           ("1e7-1e8", 1e7, 1e8), ("1e8-1e9", 1e8, 1e9),
           ("le_1e9", 0, 1e9 + 1)]
CIDX = {"twin": 0, "cousin": 1, "sexy": 2}
WIDX = {w: i for i, (w, _, _) in enumerate(WINDOWS)}


def prereg_gate():
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", PREREG],
                             cwd=ROOT, capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} is "
                 f"{'not committed to git' if not tracked else 'modified since commit'}.\n"
                 "Commit the preregistered predictions first (see 03_commit_predictions.py).")


def cell_features(g, rho, hazard_grid, small, mu_cell):
    """Per-observation feature matrix; every statistic = f(weighted col sums)."""
    cols = [g.astype(np.float64), rho, rho**2, rho**3, rho**4]
    cols += [(rho > t * mu_cell).astype(np.float64) for t in TAILS]
    cols += [(g == v).astype(np.float64) for v in small]
    for v in hazard_grid:
        cols.append((g == v).astype(np.float64))
        cols.append((g >= v).astype(np.float64))
    return np.column_stack(cols)


def stats_from_sums(s, nb, n_small, n_hz):
    """Map feature sums -> statistic vector (order fixed; see labels())."""
    out = []
    mean_g = s[..., 0] / nb
    m1 = s[..., 1] / nb
    m2 = s[..., 2] / nb
    m3 = s[..., 3] / nb
    m4 = s[..., 4] / nb
    var = m2 - m1**2
    sd = np.sqrt(np.maximum(var, 1e-300))
    c3 = m3 - 3 * m1 * m2 + 2 * m1**3
    c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    out += [mean_g, var, c3 / sd**3, c4 / sd**4 - 3.0]
    k = 5
    for _ in TAILS:
        out.append(s[..., k] / nb); k += 1
    for _ in range(n_small):
        out.append(s[..., k] / nb); k += 1
    for _ in range(n_hz):
        eq, geq = s[..., k], s[..., k + 1]
        out.append(np.where(geq > 0, eq / np.maximum(geq, 1), np.nan))
        k += 2
    return np.stack(out, axis=-1)  # (..., nstats)


def labels(small, hazard_grid):
    lab = ["mean_gap_raw", "var_norm_gap", "skew_norm_gap", "kurt_norm_gap"]
    lab += [f"p_tail_{t}x" for t in TAILS]
    lab += [f"p_gap_{v}" for v in small]
    lab += [("hazard", int(v)) for v in hazard_grid]
    return lab


def bootstrap_ses(F, seed_key, n_small, n_hz, extra_pairs=None):
    """Poisson-bootstrap SEs of all statistics derived from F's column sums.

    Returns (se_vector, autocorr_se) — autocorr from pair bootstrap if
    extra_pairs=(x, y) given.
    """
    n, nf = F.shape
    rng = np.random.default_rng(seed_key)
    sums = np.empty((B, nf))
    nbs = np.empty(B)
    chunk = max(1, min(B, int(4e7 // max(n, 1))))
    done = 0
    while done < B:
        m = min(chunk, B - done)
        w = rng.poisson(1.0, (m, n)).astype(np.float64)
        sums[done:done + m] = w @ F
        nbs[done:done + m] = w.sum(axis=1)
        done += m
    boot = stats_from_sums(sums, nbs, n_small, n_hz)
    se = np.nanstd(boot, axis=0, ddof=1)

    ac_se = float("nan")
    if extra_pairs is not None and len(extra_pairs[0]) > 3:
        x, y = extra_pairs
        Fp = np.column_stack([x, y, x * x, y * y, x * y])
        npair = len(x)
        sums_p = np.empty((B, 5))
        nbp = np.empty(B)
        chunk = max(1, min(B, int(4e7 // max(npair, 1))))
        done = 0
        while done < B:
            m = min(chunk, B - done)
            w = rng.poisson(1.0, (m, npair)).astype(np.float64)
            sums_p[done:done + m] = w @ Fp
            nbp[done:done + m] = w.sum(axis=1)
            done += m
        mx, my = sums_p[:, 0] / nbp, sums_p[:, 1] / nbp  # nbp shape (B,)
        vx = sums_p[:, 2] / nbp - mx**2
        vy = sums_p[:, 3] / nbp - my**2
        cxy = sums_p[:, 4] / nbp - mx * my
        with np.errstate(invalid="ignore", divide="ignore"):
            ac = cxy / np.sqrt(vx * vy)
        ac_se = float(np.nanstd(ac, ddof=1))
    return se, ac_se


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cx, cy = x - mx, y - my
    d = math.sqrt((cx @ cx) * (cy @ cy))
    return float((cx @ cy) / d) if d > 0 else float("nan")


def main():
    published_only = "--published-only" in sys.argv
    if not published_only:
        prereg_gate()

    t0 = time.time()
    out = {k: [] for k in ["constellation", "modulus", "class", "window",
                           "statistic", "g", "n_obs", "empirical", "se"]}

    def emit(name, mod, r, win, stat, g, n, val, se):
        out["constellation"].append(name); out["modulus"].append(mod)
        out["class"].append(r); out["window"].append(win)
        out["statistic"].append(stat); out["g"].append(g)
        out["n_obs"].append(n); out["empirical"].append(val); out["se"].append(se)

    constellations = {"twin": 2} if published_only else hz.CONSTELLATIONS
    for name, c in constellations.items():
        tbl = pq.read_table(os.path.join(HERE, "data", f"pairs_{name}.parquet"))
        P = tbl["p"].to_numpy()
        G = tbl["gap"].to_numpy().astype(np.int64)
        S = hz.S_const(c)
        RHO_ALL = G * S / np.log(P) ** 2
        small = [6, 12, 18, 30] if c in (2, 4) else [2, 4, 6, 12, 18, 30]
        hazard_grid = ([6 * m for m in
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 24, 29, 35, 42, 50, 60]])
        if c == 6:
            hazard_grid = [2, 4] + hazard_grid
        lab = labels(small, hazard_grid)

        for mod in MODULI:
            classes = hz.valid_classes(mod, c)
            RES = P % mod
            for win, lo, hi in WINDOWS:
                if published_only and win != "le_1e9":
                    continue
                wmask = (P >= lo) & (P < hi) if win != "le_1e9" else np.ones(len(P), bool)
                cell_stats, cell_ses, cell_ns, cell_ac = {}, {}, {}, {}
                cell_nrisk = {}
                for r in classes:
                    sel = np.flatnonzero(wmask & (RES == r))
                    g, rho = G[sel], RHO_ALL[sel]
                    n = len(g)
                    cell_ns[r] = n
                    if n < 5:
                        cell_stats[r] = np.full(len(lab), np.nan)
                        cell_ses[r] = np.full(len(lab), np.nan)
                        cell_ac[r] = (float("nan"), float("nan"))
                        cell_nrisk[r] = [int((g >= v).sum()) for v in hazard_grid]
                        continue
                    mu_cell = float(rho.mean())
                    F = cell_features(g, rho, hazard_grid, small, mu_cell)
                    point = stats_from_sums(F.sum(axis=0)[None, :],
                                            np.array([float(n)]),
                                            len(small), len(hazard_grid))[0]
                    if published_only:
                        se = np.full(len(lab), np.nan)
                        ac, ac_se = float("nan"), float("nan")
                    else:
                        se, ac_se = bootstrap_ses(
                            F, (SEED, CIDX[name], mod, WIDX[win], r),
                            len(small), len(hazard_grid),
                            extra_pairs=(rho[:-1], rho[1:]))
                        ac = pearson(rho[:-1], rho[1:])
                    cell_stats[r] = point
                    cell_ses[r] = se
                    cell_ac[r] = (ac, ac_se)
                    cell_nrisk[r] = [int(x) for x in
                                     F[:, 5 + len(TAILS) + len(small) + 1::2].sum(axis=0)]

                grand = np.nanmean([cell_stats[r][0] for r in classes])
                for r in classes:
                    st, se, n = cell_stats[r], cell_ses[r], cell_ns[r]
                    emit(name, mod, r, win, "mean_norm_gap", None, n,
                         st[0] / grand, se[0] / grand)
                    hz_i = 0
                    for i, l in enumerate(lab[1:], start=1):
                        if isinstance(l, tuple):  # hazard row
                            emit(name, mod, r, win, "hazard", l[1],
                                 cell_nrisk[r][hz_i], st[i], se[i])
                            hz_i += 1
                        else:
                            emit(name, mod, r, win, l, None, n, st[i], se[i])
                    emit(name, mod, r, win, "autocorr_lag1", None,
                         max(n - 1, 0), cell_ac[r][0], cell_ac[r][1])
                print(f"{name} mod {mod} {win}: {len(classes)} classes "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- regression test vs published paper numbers ----------------
    import csv as _csv
    def check(mod, fname, key):
        pub = {int(row[key]): (float(row["rho_bar"]), float(row["se_rho_bar"]))
               for row in _csv.DictReader(open(os.path.join(
                   ROOT, "project", "data", "residue", fname)))
               if row.get("xmax", "1e+09") in ("1e+09", None) or "xmax" not in row}
        mine = {}
        for i in range(len(out["class"])):
            if (out["constellation"][i] == "twin" and out["modulus"][i] == mod
                    and out["window"][i] == "le_1e9"
                    and out["statistic"][i] == "mean_norm_gap"):
                mine[out["class"][i]] = out["empirical"][i]
        diffs = [abs(mine[r] - pub[r][0]) for r in pub]
        assert max(diffs) < 5e-6, f"mod {mod}: empirical rho_bar mismatch {max(diffs):.2e}"
        print(f"REGRESSION TEST mod {mod}: {len(pub)} classes reproduce published "
              f"rho_bar exactly (max diff {max(diffs):.1e})")

    check(210, "rho_bar_mod210.csv", "mod210")
    check(2310, "rho_bar_mod2310.csv", "mod2310")

    if published_only:
        print("published-only smoke test PASSED; no output written")
        return

    tbl = pa.table({
        "constellation": pa.array(out["constellation"], pa.string()),
        "modulus": pa.array(out["modulus"], pa.int32()),
        "class": pa.array(out["class"], pa.int32()),
        "window": pa.array(out["window"], pa.string()),
        "statistic": pa.array(out["statistic"], pa.string()),
        "g": pa.array(out["g"], pa.int32()),
        "n_obs": pa.array(out["n_obs"], pa.int64()),
        "empirical": pa.array(out["empirical"], pa.float64()),
        "se": pa.array(out["se"], pa.float64()),
    })
    pq.write_table(tbl, OUT)
    print(f"wrote {OUT}: {tbl.num_rows:,} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
