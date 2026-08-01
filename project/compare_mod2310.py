"""
Compare preregistered mod-2310 predictions against empirical rho_bar_r,
and regenerate Figure 1 (predicted vs. empirical deviation scatter,
mod 210 + mod 2310).

Inputs (produced by predict_rho_bar_mod2310.py, rho_bar_mod2310.py, and the
original mod-210 pipeline):
    data/residue/pred_rho_bar_mod2310.csv
    data/residue/rho_bar_mod2310_cutoffs.csv
    data/residue/pred_rho_bar_mod210.csv
    data/residue/rho_bar_mod210.csv

Usage:  python compare_mod2310.py
Output: stats on stdout
        data/figures/fig1_pred_vs_emp.png / .pdf
"""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "data", "residue")
FIGDIR = os.path.join(HERE, "data", "figures")

# validated categorical palette, slots 1-2 (all-pairs pass, light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def stats(pairs):
    """pairs: list of (pred, emp, se). Returns dict of comparison stats."""
    n = len(pairs)
    xs = [p - 1 for p, _, _ in pairs]
    ys = [e - 1 for _, e, _ in pairs]
    chi2 = sum(((e - p) / se) ** 2 for p, e, se in pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r2 = sxy * sxy / (sxx * syy)
    slope = sxy / sxx

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for j, i in enumerate(order):
            rk[i] = j
        return rk

    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho_s = 1 - 6 * d2 / (n * (n * n - 1))
    return dict(n=n, chi2=chi2, r2=r2, slope=slope, spearman=rho_s,
                spread_emp=max(ys) - min(ys), spread_pred=max(xs) - min(xs))


def main():
    pred2310 = read_csv("pred_rho_bar_mod2310.csv")
    emp2310 = read_csv("rho_bar_mod2310_cutoffs.csv")
    pred210 = read_csv("pred_rho_bar_mod210.csv")
    emp210 = read_csv("rho_bar_mod210.csv")

    pred = {(int(r["mod2310"]), r["xmax"]):
            (float(r["pred_rho_bar"]),
             float(r["pred_rho_bar_wheelonly"]) if r["pred_rho_bar_wheelonly"] else None)
            for r in pred2310}
    emp = {(int(r["mod2310"]), r["xmax"]): (float(r["rho_bar"]), float(r["se_rho_bar"]))
           for r in emp2310}
    cutoffs = sorted({x for _, x in emp}, key=float)

    # ---- main comparison at largest cutoff ----
    xmain = cutoffs[-1]
    classes = sorted({r for r, x in emp if x == xmain})
    main_pairs = [(pred[(r, xmain)][0],) + emp[(r, xmain)] for r in classes]
    s = stats(main_pairs)
    wheel_pairs = [(pred[(r, xmain)][1],) + emp[(r, xmain)] for r in classes]
    sw = stats(wheel_pairs)

    print(f"=== mod 2310, {len(classes)} classes, x <= {xmain} ===")
    print(f"full model : chi2 = {s['chi2']:.0f} (dof={s['n']})  r^2 = {s['r2']:.4f}  "
          f"slope = {s['slope']:.3f}  Spearman = {s['spearman']:.4f}")
    print(f"wheel-only : chi2 = {sw['chi2']:.0f} (dof={sw['n']})  r^2 = {sw['r2']:.4f}  "
          f"slope = {sw['slope']:.3f}  Spearman = {sw['spearman']:.4f}")
    print(f"deviation spread: empirical {s['spread_emp']:.4f}, predicted {s['spread_pred']:.4f}")

    resid = sorted(((emp[(r, xmain)][0] - pred[(r, xmain)][0], r) for r in classes),
                   key=lambda t: -abs(t[0]))
    print("largest residuals:", [(r, f"{d:+.4f}") for d, r in resid[:6]])
    nsig = sum(1 for r in classes
               if abs(emp[(r, xmain)][0] - pred[(r, xmain)][0]) / emp[(r, xmain)][1] > 3)
    print(f"classes with |z| > 3: {nsig} of {len(classes)}")

    # ---- 1/log x decay check across cutoffs ----
    print(f"\n=== 1/log x decay check ===")
    print(f"{'xmax':>7} {'spread_emp':>11} {'spread_pred':>12} {'r^2':>8} {'slope':>7}")
    for x in cutoffs:
        cls = sorted({r for r, xx in emp if xx == x})
        sx = stats([(pred[(r, x)][0],) + emp[(r, x)] for r in cls])
        print(f"{x:>7} {sx['spread_emp']:>11.4f} {sx['spread_pred']:>12.4f} "
              f"{sx['r2']:>8.4f} {sx['slope']:>7.3f}")

    # ---- Figure 1: predicted vs empirical deviation scatter ----
    p210 = {int(r["mod210"]): float(r["pred_rho_bar"]) for r in pred210}
    e210 = {int(r["mod210"]): (float(r["rho_bar"]), float(r["se_rho_bar"])) for r in emp210}
    s210 = stats([(p210[r],) + e210[r] for r in p210])

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lim = 0.16
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="#b5b4ad", zorder=1)

    xs = [pred[(r, xmain)][0] - 1 for r in classes]
    ys = [emp[(r, xmain)][0] - 1 for r in classes]
    es = [emp[(r, xmain)][1] for r in classes]
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=BLUE, elinewidth=0.7,
                alpha=0.35, zorder=2)
    ax.scatter(xs, ys, s=16, color=BLUE, alpha=0.75, linewidths=0, zorder=3,
               label=f"mod 2310 (135 classes), $r^2$ = {s['r2']:.3f}")

    x210 = [p210[r] - 1 for r in sorted(p210)]
    y210 = [e210[r][0] - 1 for r in sorted(p210)]
    ax.scatter(x210, y210, s=52, facecolors="none", edgecolors=ORANGE,
               linewidths=1.6, zorder=4,
               label=f"mod 210 (15 classes), $r^2$ = {s210['r2']:.3f}")

    ax.set_xlabel(r"predicted deviation  $\bar\rho_r - 1$", color=INK)
    ax.set_ylabel(r"empirical deviation  $\bar\rho_r - 1$", color=INK)
    ax.set_title("Class-conditional twin-prime gap deviations:\n"
                 "pair-correlation prediction vs. sieve data ($x \\leq 10^9$)",
                 fontsize=11, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.grid(True, lw=0.4, color="#e8e7e2", zorder=0)
    ax.text(lim * 0.95, -lim * 0.92, "y = x", color=INK2, fontsize=8,
            ha="right", style="italic")

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGDIR, f"fig1_pred_vs_emp.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
