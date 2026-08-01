"""
Class-conditional mean gaps between consecutive twin primes,
derived from the Hardy-Littlewood pair-correlation (4-tuple) singular series.

Model:
  Given a twin prime with first element p == r (mod 210), the conditional
  intensity of a twin at p+g (per integer, in a window where log x = L) is

    q_r(g) = (S2 / L^2) * WHEEL_r(g) * C(g)

  where
    WHEEL_r(g) = prod over p0 in {2,3,5,7} of
                 indicator[(r+g) and (r+g+2) are nonzero mod p0] * p0/(p0 - nu2(p0))
                 with nu2(p0) = #distinct residues of {0,2} mod p0
                 (this is the deterministic wheel factor: p == r mod 210 is FIXED)
    C(g)       = prod over primes p > 7 of  p*(p - nu4(p,g)) / (p-2)^2
                 with nu4 = #distinct residues of {0,2,g,g+2} mod p
                 (nu4 = 4 generically; 2 if p|g; 3 if p|(g-2) or p|(g+2))
    S2         = twin prime constant * 2 = prod_{p>2} (1 - 1/(p-1)^2) ... = 1.32032...

  Next-gap distribution (discrete hazard over even g):
    P_r(g) = q_r(g) * prod_{h < g, h even} (1 - q_r(h))
  Conditional mean gap:  M_r = sum_g g * P_r(g)
  Normalized:            rho_r = M_r / mean_r'(M_r')   (equidistribution over classes)

Also computed: a "wheel-only" null (C(g) := const) to separate pure wheel
geometry from genuine singular-series correlation structure.
"""

import math
from sympy import primerange, factorint

WHEEL = 210
WHEEL_PRIMES = [2, 3, 5, 7]
S2 = 2 * 0.6601618158468696  # 2 * C_2

# ---------- valid classes ----------
def valid_classes(mod=WHEEL):
    out = []
    for r in range(mod):
        if math.gcd(r, mod) == 1 and math.gcd(r + 2, mod) == 1:
            out.append(r)
    return out

CLASSES = valid_classes()
assert len(CLASSES) == 15, len(CLASSES)

# ---------- generic tail constant for C(g) ----------
# prod over primes 7 < p <= P of p(p-4)/(p-2)^2, extended with Euler-product tail.
def generic_constant(pmax=2_000_000):
    logc = 0.0
    for p in primerange(11, pmax):
        logc += math.log(p * (p - 4)) - 2 * math.log(p - 2)
    # tail estimate: log[p(p-4)/(p-2)^2] ~ -4/p^2, sum_{p>P} ~ negligible at 2e6
    return math.exp(logc)

C_GENERIC = generic_constant()

# ---------- correction factor for a specific gap g ----------
def C_of_g(g):
    """prod_{p>7} p(p-nu4)/(p-2)^2, via corrections to the generic product."""
    c = C_GENERIC
    seen = set()
    for base, delta in ((g, 0), (g - 2, 0), (g + 2, 0)):
        for p in factorint(base):
            if p <= 7 or p in seen:
                continue
            seen.add(p)
            if g % p == 0:
                nu4 = 2
            elif (g - 2) % p == 0 or (g + 2) % p == 0:
                nu4 = 3
            else:
                continue
            generic = p * (p - 4) / (p - 2) ** 2
            actual = p * (p - nu4) / (p - 2) ** 2
            c *= actual / generic
    return c

# ---------- wheel factor ----------
def wheel_factor(r, g):
    f = 1.0
    for p0 in WHEEL_PRIMES:
        nu2 = len({0 % p0, 2 % p0})
        s = (r + g) % p0
        if s == 0 or (s + 2) % p0 == 0:
            return 0.0
        f *= p0 / (p0 - nu2)
    return f

# ---------- hazard model ----------
def class_mean_gap(r, L, gmax_mult=12, wheel_only=False):
    """Mean next-twin gap for class r at scale L = log x."""
    base = S2 / (L * L)
    mean_gap_est = L * L / S2
    gmax = int(gmax_mult * mean_gap_est) // 2 * 2
    surv = 1.0
    mean = 0.0
    total_p = 0.0
    for g in range(6, gmax + 2, 2):
        wf = wheel_factor(r, g)
        if wf == 0.0:
            continue
        cg = 1.0 if wheel_only else C_of_g(g)
        q = base * wf * cg
        if q >= 1.0:
            q = 1.0
        pg = surv * q
        mean += g * pg
        total_p += pg
        surv *= (1.0 - q)
        if surv < 1e-12:
            break
    # renormalize tiny truncation remainder
    return mean / total_p if total_p > 0 else float("nan")

def run(L, wheel_only=False):
    means = {r: class_mean_gap(r, L, wheel_only=wheel_only) for r in CLASSES}
    grand = sum(means.values()) / len(means)
    return {r: means[r] / grand for r in CLASSES}, grand

if __name__ == "__main__":
    for label, L in [("x = 1e10", math.log(1e10)), ("x = 1e12", math.log(1e12))]:
        rho_full, grand_full = run(L, wheel_only=False)
        rho_wheel, grand_wheel = run(L, wheel_only=True)
        print(f"\n=== scale {label}  (L = {L:.3f}) ===")
        print(f"grand mean gap: full model {grand_full:.1f} | wheel-only {grand_wheel:.1f}")
        print(f"{'r':>5} {'rho_full':>10} {'dev_full(%)':>12} {'rho_wheel':>10} {'dev_wheel(%)':>13} {'corr_part(%)':>13}")
        for r in CLASSES:
            df = (rho_full[r] - 1) * 100
            dw = (rho_wheel[r] - 1) * 100
            print(f"{r:>5} {rho_full[r]:>10.5f} {df:>12.3f} {rho_wheel[r]:>10.5f} {dw:>13.3f} {df - dw:>13.3f}")
        spread_f = (max(rho_full.values()) - min(rho_full.values())) * 100
        spread_w = (max(rho_wheel.values()) - min(rho_wheel.values())) * 100
        print(f"max spread: full {spread_f:.3f}%  wheel-only {spread_w:.3f}%")
