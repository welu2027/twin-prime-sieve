"""
Stage 5: join predicted + empirical into ledger/grand_ledger.csv, compute
z-scores, generate the embedding `text` column and the class-arithmetic
feature columns, and write a summary report + environment record.

Run after 04_empirical.py.
"""

import csv
import math
import os
import platform
import sys
import time

import numpy as np
import pyarrow.parquet as pq

import hazard as hz

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "grand_ledger.csv")
REPORT = os.path.join(HERE, "REPORT.md")
ENVMD = os.path.join(HERE, "ENVIRONMENT.md")

LONG_210 = {29, 41, 59}
SHORT_210 = {137, 149, 167, 179}
CNAME = {"twin": "Twin primes", "cousin": "Cousin primes (p, p+4)",
         "sexy": "Sexy primes (p, p+6)"}
WNAME = {"le_1e9": "p <= 1e9", "1e5-1e6": "window 1e5-1e6",
         "1e6-1e7": "window 1e6-1e7", "1e7-1e8": "window 1e7-1e8",
         "1e8-1e9": "window 1e8-1e9"}


def dist_to_mult(r, p):
    m = r % p
    return min(m, p - m)


def fmt(v, nd=4):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}g}"


def make_text(row):
    name = CNAME[row["constellation"]]
    where = f"class {row['class']} mod {row['modulus']}, {WNAME[row['window']]}"
    st, emp, pred, z = row["statistic"], row["empirical"], row["predicted"], row["z"]
    if emp is None or math.isnan(emp):
        return f"{name}, {where}: too few observations ({row['n_obs']})."
    ztxt = f"z = {z:+.1f}" if z is not None and not math.isnan(z) else "z undefined"
    if st == "mean_norm_gap":
        d_e, d_p = (emp - 1) * 100, (pred - 1) * 100
        word = lambda d: f"{abs(d):.1f}% {'long' if d >= 0 else 'short'}"
        return (f"{name}, {where}: mean normalized gap runs {word(d_e)}; "
                f"model predicts {word(d_p)}; {ztxt}.")
    if st == "var_norm_gap":
        lab = "variance of the normalized gap"
    elif st == "skew_norm_gap":
        lab = "skewness of the normalized gap"
    elif st == "kurt_norm_gap":
        lab = "excess kurtosis of the normalized gap"
    elif st.startswith("p_tail_"):
        lab = f"P(gap > {st[7:-1]}x class mean)"
    elif st.startswith("p_gap_"):
        lab = f"P(gap = {st[6:]})"
    elif st == "hazard":
        lab = f"hazard of the next pair at gap {row['g']}"
    elif st == "autocorr_lag1":
        return (f"{name}, {where}: lag-1 autocorrelation of consecutive "
                f"normalized gaps is {emp:+.4f} (hazard model predicts 0); {ztxt}.")
    else:
        lab = st
    return f"{name}, {where}: {lab} is {emp:.4g} vs predicted {pred:.4g}; {ztxt}."


def key(d, i):
    g = d["g"][i]
    return (d["constellation"][i], d["modulus"][i], d["class"][i],
            d["window"][i], d["statistic"][i], -1 if g is None else g)


def lin_stats(pairs):
    """pairs of (pred, emp): r^2 and slope of emp on pred."""
    xs = np.array([p for p, _ in pairs])
    ys = np.array([e for _, e in pairs])
    ok = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan"), float("nan"), len(xs)
    r = np.corrcoef(xs, ys)[0, 1]
    slope = np.cov(xs, ys)[0, 1] / np.var(xs)
    return r * r, slope, len(xs)


def main():
    t0 = time.time()
    pred = pq.read_table(os.path.join(HERE, "data", "predicted.parquet")).to_pydict()
    emp = pq.read_table(os.path.join(HERE, "data", "empirical.parquet")).to_pydict()

    pmap = {key(pred, i): i for i in range(len(pred["class"]))}
    n_rows = len(emp["class"])

    cols = ["constellation", "constellation_pattern", "modulus", "class",
            "window", "statistic", "g", "n_obs", "empirical", "predicted",
            "predicted_wheel_only", "se", "z", "z_wheel_only",
            "class_mod_6", "class_mod_30", "class_mod_7", "class_mod_11",
            "dist_to_mult_7", "dist_to_mult_11", "paper_direction", "text"]

    stat_pairs = {}     # (constellation, mod, statistic) -> [(pred, emp)] at le_1e9
    spread_rows = {}    # (constellation, mod, window) -> [(pred, emp)] mean only
    z_extreme = []
    n_z3 = 0
    n_z_valid = 0

    with open(LEDGER, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n_rows):
            k = key(emp, i)
            j = pmap.get(k)
            assert j is not None, f"no prediction for cell {k}"
            row = {c: emp[c][i] for c in
                   ["constellation", "modulus", "class", "window",
                    "statistic", "g", "n_obs", "empirical", "se"]}
            row["constellation_pattern"] = pred["constellation_pattern"][j]
            row["predicted"] = pred["predicted"][j]
            row["predicted_wheel_only"] = pred["predicted_wheel_only"][j]
            e, p, pw, se = (row["empirical"], row["predicted"],
                            row["predicted_wheel_only"], row["se"])
            valid = (e is not None and not math.isnan(e)
                     and se is not None and not math.isnan(se) and se > 0)
            row["z"] = (e - p) / se if valid else float("nan")
            row["z_wheel_only"] = (e - pw) / se if valid else float("nan")
            r = row["class"]
            row["class_mod_6"] = r % 6
            row["class_mod_30"] = r % 30
            row["class_mod_7"] = r % 7
            row["class_mod_11"] = r % 11
            row["dist_to_mult_7"] = dist_to_mult(r, 7)
            row["dist_to_mult_11"] = dist_to_mult(r, 11)
            row["paper_direction"] = ""
            if row["constellation"] == "twin" and row["modulus"] == 210:
                row["paper_direction"] = ("long" if r in LONG_210 else
                                          "short" if r in SHORT_210 else "neutral")
            row["text"] = make_text(row)
            w.writerow([
                row["constellation"], row["constellation_pattern"],
                row["modulus"], row["class"], row["window"], row["statistic"],
                "" if row["g"] is None else row["g"], row["n_obs"],
                fmt(e, 8), fmt(p, 8), fmt(pw, 8), fmt(se, 6),
                fmt(row["z"], 4), fmt(row["z_wheel_only"], 4),
                row["class_mod_6"], row["class_mod_30"], row["class_mod_7"],
                row["class_mod_11"], row["dist_to_mult_7"],
                row["dist_to_mult_11"], row["paper_direction"], row["text"]])

            if valid:
                n_z_valid += 1
                if abs(row["z"]) > 3:
                    n_z3 += 1
                    z_extreme.append((abs(row["z"]), row["constellation"],
                                      row["modulus"], r, row["window"],
                                      row["statistic"], row["g"], row["z"]))
            if row["window"] == "le_1e9":
                stat_pairs.setdefault(
                    (row["constellation"], row["modulus"], row["statistic"]),
                    []).append((p, e))
            if row["statistic"] == "mean_norm_gap":
                spread_rows.setdefault(
                    (row["constellation"], row["modulus"], row["window"]),
                    []).append((p, e))

    size_mb = os.path.getsize(LEDGER) / 1e6
    assert size_mb < 50, f"grand_ledger.csv is {size_mb:.1f} MB (cap 50)"

    # ---------------- report ----------------
    lines = ["# Grand Ledger — summary report", ""]
    lines.append(f"Rows: {n_rows:,}; file size {size_mb:.1f} MB; "
                 f"cells with defined z: {n_z_valid:,}; |z| > 3: {n_z3:,} "
                 f"({100*n_z3/max(n_z_valid,1):.2f}%; ~0.27% expected under the model)")
    lines.append("")
    lines.append("## r² and slope of empirical vs predicted, per statistic (window le_1e9)")
    lines.append("")
    lines.append("| constellation | modulus | statistic | n | r² | slope |")
    lines.append("|---|---|---|---|---|---|")
    for (cn, mod, st), pairs in sorted(stat_pairs.items()):
        r2, slope, n = lin_stats(pairs)
        if not math.isnan(r2):
            lines.append(f"| {cn} | {mod} | {st} | {n} | {r2:.4f} | {slope:.3f} |")
    lines.append("")
    lines.append("## 1/log x check — mean_norm_gap deviation spread per window")
    lines.append("")
    lines.append("| constellation | modulus | window | spread_emp | spread_pred | r² |")
    lines.append("|---|---|---|---|---|---|")
    for (cn, mod, win), pairs in sorted(spread_rows.items()):
        xs = [p for p, e in pairs if not math.isnan(e)]
        ys = [e for p, e in pairs if not math.isnan(e)]
        if len(ys) < 3:
            continue
        r2, _, _ = lin_stats(pairs)
        lines.append(f"| {cn} | {mod} | {win} | {max(ys)-min(ys):.4f} | "
                     f"{max(xs)-min(xs):.4f} | {r2:.4f} |")
    lines.append("")
    lines.append("## largest |z| cells")
    lines.append("")
    for a, cn, mod, r, win, st, g, z in sorted(z_extreme, reverse=True)[:20]:
        gtxt = f" g={g}" if g is not None and g >= 0 else ""
        lines.append(f"- {cn} mod {mod} class {r} {win} {st}{gtxt}: z = {z:+.2f}")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(ENVMD, "w") as f:
        import pyarrow
        f.write(f"""# Ledger build environment

- date: 2026-08-27
- python: {platform.python_version()} ({platform.platform()})
- numpy: {np.__version__}
- pyarrow: {pyarrow.__version__}
- bootstrap: Poisson bootstrap, B = 1000, seed = (20260827, constellation_idx,
  modulus, window_idx, class); autocorr via pair bootstrap, same seed stream
- sieve: odd-only segmented numpy sieve (ledger/01_sieve.py), XMAX = 1e9
- hazard model: ledger/hazard.py; Euler products truncated at p < 2e6;
  gmax = 12x predicted mean gap; window nodes weighted e^u/u^2
- preregistration: ledger/prereg/predictions_20260827.csv (SHA-256 printed by
  03_commit_predictions.py; committed to git before stage 4 ran)
""")

    print(f"wrote {LEDGER} ({size_mb:.1f} MB)")
    print(f"wrote {REPORT}")
    print(f"wrote {ENVMD}")
    print(f"total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
