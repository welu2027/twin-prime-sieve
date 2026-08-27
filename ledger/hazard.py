"""
Generalized class-conditional hazard model for prime constellations (0, c),
c in {2 (twin), 4 (cousin), 6 (sexy)}.

Extends project/class_conditional_gaps.py (twins only) to arbitrary even c.
Model, for a constellation pair with first element p == r (mod M), M a
primorial wheel (210 or 2310):

    q_r(g) = (S_c / L^2) * WHEEL_r(g) * C(g),   L = log x

  WHEEL_r(g) = prod over wheel primes p0 | M of
               1{(r+g) !== 0 and (r+g+c) !== 0 mod p0} * p0/(p0 - nu2(p0))
               nu2(p0) = #distinct residues of {0, c} mod p0
  C(g)       = prod over primes p > wheel_max of p*(p - nu4)/(p-2)^2
               nu4(p,g) = #distinct residues of {0, c, g, g+c} mod p
                        = 2 if p|g; 3 if p|(g-c) or p|(g+c); else 4
               (valid for p > 7 since every prime dividing c is a wheel prime;
                the generic p(p-4)/(p-2)^2 and the correction machinery are
                therefore IDENTICAL to the twin case with 2 -> c)
  S_c        = 2*C2 * prod_{p | c, p > 2} (p-1)/(p-2)
               (2*C2 for twins/cousins, 4*C2 for sexy)

Overlap special case (sexy only): the gap g = c is admissible (consecutive
sexy firsts can differ by 6, chaining (p, p+6), (p+6, p+12)).  There the
"4-tuple" degenerates to the 3-tuple {0, c, 2c} and the event is a SINGLE new
prime p + 2c, so the hazard is order 1/L, not 1/L^2:

    q_r(c) = (1/L) * WHEEL1_r(2c) * K3
  WHEEL1_r(2c) = prod over wheel primes of 1{(r+2c) !== 0 mod p0} * p0/(p0-1)
  K3           = prod_{p > wheel_max} p(p-3)/((p-2)(p-1))   (convergent)

(The naive 4-tuple Euler product diverges at g = c; that divergence is the
formalism signalling the change of order in 1/L.)  In the wheel-only null
(C == 1) the same expression is used with K3 := 1.

Discrete hazard chain over the constellation's admissible gap support
(multiples of 6 for twins/cousins; even g >= 2 for sexy):
    P_r(g) = q_r(g) * prod_{h < g} (1 - q_r(h)),  renormalized.

All heavy paths are numpy-vectorized; factorization uses a smallest-prime-
factor table (no sympy dependency).
"""

import math
from functools import lru_cache

import numpy as np

C2 = 0.6601618158468696  # twin prime constant
PMAX_PRODUCT = 2_000_000  # Euler products truncated here (matches old code)

CONSTELLATIONS = {"twin": 2, "cousin": 4, "sexy": 6}


# ---------------------------------------------------------------- primes / spf
@lru_cache(maxsize=None)
def _primes_below(n):
    s = np.ones(n, dtype=bool)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p:: p] = False
    return np.flatnonzero(s)


@lru_cache(maxsize=None)
def _spf_table(n):
    """smallest prime factor for 0..n-1"""
    spf = np.zeros(n, dtype=np.int64)
    for p in range(2, int(n**0.5) + 1):
        if spf[p] == 0:
            spf[p::p][spf[p::p] == 0] = p
    rest = np.flatnonzero(spf == 0)
    spf[rest] = rest  # primes (and 0,1 map to themselves)
    return spf


def _prime_factors(n, spf):
    out = []
    n = abs(int(n))
    while n > 1:
        p = int(spf[n])
        out.append(p)
        while n % p == 0:
            n //= p
    return out


# ---------------------------------------------------------------- constants
def wheel_primes_of(mod):
    ps = [p for p in (2, 3, 5, 7, 11, 13) if mod % p == 0]
    assert math.prod(ps) == mod, f"{mod} is not a primorial"
    return ps


def S_const(c):
    s = 2 * C2
    for p in set(_prime_factors(c, _spf_table(100))):
        if p > 2:
            s *= (p - 1) / (p - 2)
    return s


@lru_cache(maxsize=None)
def generic_constant(pmin):
    """prod_{pmin <= p < 2e6} p(p-4)/(p-2)^2 (tail beyond 2e6 negligible)."""
    ps = _primes_below(PMAX_PRODUCT)
    ps = ps[ps >= pmin].astype(np.float64)
    return float(np.exp(np.sum(np.log(ps * (ps - 4)) - 2 * np.log(ps - 2))))


@lru_cache(maxsize=None)
def K3_constant(wheel_max):
    """prod_{p > wheel_max} p(p-3)/((p-2)(p-1)) for the g = c overlap case."""
    ps = _primes_below(PMAX_PRODUCT)
    ps = ps[ps > wheel_max].astype(np.float64)
    return float(np.exp(np.sum(np.log(ps) + np.log(ps - 3)
                               - np.log(ps - 2) - np.log(ps - 1))))


_NEXT_PRIME = {7: 11, 11: 13}


# ---------------------------------------------------------------- classes
def valid_classes(mod, c):
    return [r for r in range(mod)
            if math.gcd(r, mod) == 1 and math.gcd(r + c, mod) == 1]


def gap_support(c, gmax):
    """Admissible inter-first gaps: multiples of 6 for twin/cousin
    (firsts occupy a single class mod 6), even g >= 2 for sexy."""
    if c in (2, 4):
        return np.arange(6, gmax + 1, 6, dtype=np.int64)
    return np.arange(2, gmax + 1, 2, dtype=np.int64)


# ---------------------------------------------------------------- C(g) vector
@lru_cache(maxsize=None)
def _C_array_cached(c, wheel_max, gmax):
    gs = gap_support(c, gmax)
    base = generic_constant(_NEXT_PRIME[wheel_max])
    spf = _spf_table(int(gmax + c + 2))
    out = np.full(len(gs), base, dtype=np.float64)
    for i, g in enumerate(gs):
        g = int(g)
        if g == c:
            continue  # overlap case handled at hazard level
        seen = set()
        for b in (g, g - c, g + c):
            if b == 0:
                continue
            for p in _prime_factors(b, spf):
                if p <= wheel_max or p in seen:
                    continue
                seen.add(p)
                if g % p == 0:
                    nu4 = 2
                elif (g - c) % p == 0 or (g + c) % p == 0:
                    nu4 = 3
                else:
                    continue
                out[i] *= (p * (p - nu4) / (p - 2) ** 2) / (p * (p - 4) / (p - 2) ** 2)
    return gs, out


def C_array(c, wheel_max, gmax):
    gs, arr = _C_array_cached(c, wheel_max, int(gmax))
    return gs, arr


# ---------------------------------------------------------------- wheel factors
def wheel_vector(r, gs, c, wheel_primes):
    """WHEEL_r(g) for each g in gs (0 where inadmissible)."""
    f = np.ones(len(gs), dtype=np.float64)
    for p0 in wheel_primes:
        nu2 = len({0, c % p0})
        s = (r + gs) % p0
        bad = (s == 0) | ((s + c) % p0 == 0)
        f = np.where(bad, 0.0, f * (p0 / (p0 - nu2)))
    return f


def wheel1_factor(m, wheel_primes):
    """prod p0/(p0-1) * 1{m !== 0 mod p0} — single-prime wheel factor."""
    f = 1.0
    for p0 in wheel_primes:
        if m % p0 == 0:
            return 0.0
        f *= p0 / (p0 - 1)
    return f


# ---------------------------------------------------------------- hazard chain
def class_gap_distribution(r, L, c, mod, wheel_only=False, gmax_mult=12):
    """Full next-pair gap distribution P(g) for class r (mod `mod`) at scale L.

    Returns (gs, P) with P renormalized to sum 1 over the truncated support.
    """
    wps = wheel_primes_of(mod)
    wheel_max = wps[-1]
    S = S_const(c)
    mean_est = L * L / S
    gmax = int(gmax_mult * mean_est) // 6 * 6 + 6
    gs, Cg = C_array(c, wheel_max, gmax)
    if wheel_only:
        Cg = np.ones_like(Cg)
    q = (S / (L * L)) * wheel_vector(r, gs, c, wps) * Cg
    if c == 6:
        # overlap g = c: single new prime, order 1/L
        i = int(np.searchsorted(gs, c))
        if i < len(gs) and gs[i] == c:
            k3 = 1.0 if wheel_only else K3_constant(wheel_max)
            q[i] = wheel1_factor(r + 2 * c, wps) * k3 / L
    np.clip(q, 0.0, 1.0, out=q)
    surv = np.cumprod(1.0 - q)
    P = q * np.concatenate(([1.0], surv[:-1]))
    tot = P.sum()
    if tot <= 0:
        return gs, P
    return gs, P / tot


def class_mean_gap(r, L, c=2, mod=210, wheel_only=False, gmax_mult=12):
    gs, P = class_gap_distribution(r, L, c, mod, wheel_only, gmax_mult)
    return float((gs * P).sum())
