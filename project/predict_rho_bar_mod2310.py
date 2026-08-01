"""
PREREGISTRATION: mod-2310 predictions for class-conditional twin-prime gaps.

Extends the mod-210 analysis (class_conditional_gaps.py / compare.py) by adding
prime 11 to the wheel: 135 admissible classes r mod 2310 with
gcd(r, 2310) = gcd(r+2, 2310) = 1.

For each class this computes the predicted normalized mean next-twin gap
rho_bar_r = M_r / mean_r'(M_r'), density-weighted over the data window
[1e5, xmax] with weight e^u/u^2 (u = log x) at 13 nodes — identical scheme
to compare.py.

Run this BEFORE sieving the empirical mod-2310 gaps; the output CSV is the
preregistered prediction. Commit it before running rho_bar_mod2310.py.

Usage:  python predict_rho_bar_mod2310.py
Output: data/residue/pred_rho_bar_mod2310.csv   (135 classes x 3 cutoffs)
        data/residue/pred_rho_bar_mod210.csv    (same scheme at mod 210, for Figure 1)
"""

import csv
import math
import os
import sys
import time

import class_conditional_gaps as m

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "residue")
CUTOFFS = [1e7, 1e8, 1e9]
UMIN = 11.5   # ~1e5, matches compare.py
NODES = 13


def window(xmax):
    us = [UMIN + i * (math.log(xmax) - UMIN) / (NODES - 1) for i in range(NODES)]
    wts = [math.exp(u) / u**2 for u in us]
    return us, wts


def predict(mod, xmax, wheel_only=False):
    """Density-weighted mean gap per class over [e^UMIN, xmax], normalized."""
    classes = m.valid_classes(mod)
    us, wts = window(xmax)
    W = sum(wts)
    pred = {r: 0.0 for r in classes}
    for u, w in zip(us, wts):
        for r in classes:
            pred[r] += w * m.class_mean_gap(r, u, wheel_only=wheel_only, mod=mod)
    for r in classes:
        pred[r] /= W
    grand = sum(pred.values()) / len(classes)
    return {r: pred[r] / grand for r in classes}, grand


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    # --- mod 2310, full model at each cutoff + wheel-only null at 1e9 ---
    rows = {}
    for xmax in CUTOFFS:
        rho, grand = predict(2310, xmax)
        print(f"mod 2310, xmax={xmax:.0e}: predicted grand mean gap {grand:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for r, v in rho.items():
            rows[(r, xmax)] = [v, ""]
    rho_w, _ = predict(2310, CUTOFFS[-1], wheel_only=True)
    for r, v in rho_w.items():
        rows[(r, CUTOFFS[-1])][1] = f"{v:.6f}"

    path = os.path.join(OUTDIR, "pred_rho_bar_mod2310.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mod2310", "xmax", "pred_rho_bar", "pred_rho_bar_wheelonly"])
        for (r, xmax), (v, vw) in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            w.writerow([r, f"{xmax:.0e}", f"{v:.6f}", vw])
    print(f"wrote {path}")

    # --- mod 210 at 1e9 (same scheme), for the Figure 1 overlay ---
    rho210, _ = predict(210, CUTOFFS[-1])
    rho210_w, _ = predict(210, CUTOFFS[-1], wheel_only=True)
    path210 = os.path.join(OUTDIR, "pred_rho_bar_mod210.csv")
    with open(path210, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mod210", "xmax", "pred_rho_bar", "pred_rho_bar_wheelonly"])
        for r in m.valid_classes(210):
            w.writerow([r, f"{CUTOFFS[-1]:.0e}", f"{rho210[r]:.6f}", f"{rho210_w[r]:.6f}"])
    print(f"wrote {path210}")

    spread = max(v for v, _ in rows.values()) - min(v for v, _ in rows.values())
    print(f"predicted mod-2310 deviation spread (all cutoffs pooled): {spread:.4f}")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
