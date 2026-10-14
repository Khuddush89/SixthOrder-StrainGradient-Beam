r"""
Green's function of the sixth-order strain-gradient beam problem

    -u^{(6)}(x) + alpha^2 u^{(4)}(x) = f(x),        0 < x < L,
    u(0) = u(L) = 0,   u''(0) = u''(L) = 0,
    u'''(0) - alpha^2 u'(0) = 0,   u'''(L) - alpha^2 u'(L) = 0,

in the notation of the paper:

* ``G0``      stable Green's function of L0 u = u^(6) - alpha^2 u^(4)
              (Theorem 2.1); G0 <= 0.
* ``G0_raw``  raw coefficient expression retained only for algebra checks.
* ``Gt``      Gtilde = -G0 >= 0, so that u(x) = int_0^L Gt(x,s) f(s) ds.
* ``G2``      Dirichlet Green's function of v'' - alpha^2 v (non-positive).
* ``G4``      clamped Euler--Bernoulli kernel, w'''' = delta, w = w' = 0.
* ``Gt_conv`` the factorised representation Gt = int (-G2) G4 dxi
              (Lemma 2.1), evaluated by Gauss--Legendre quadrature.

Constants used in the existence theorem:

* ``A_const``  A = max_x int_0^L Gt(x,s) ds  (closed form, attained at L/2),
* ``B_const``  B = min_{x in [delta, L-delta]} int_delta^{L-delta} Gt(x,s) ds,
* ``eta_ratio`` r(s) = min_{[delta,L-delta]} Gt(.,s) / max_{[0,L]} Gt(.,s),
* ``eta_star`` the optimal constant eta* = inf_s r(s) of Lemma 2.3.

Author: M. Khuddush.  Licence: MIT.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_GL: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def gauss_legendre(n: int, a: float = 0.0, b: float = 1.0):
    """Gauss--Legendre nodes and weights on [a, b]."""
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    t, w = _GL[n]
    return 0.5 * (b - a) * t + 0.5 * (b + a), 0.5 * (b - a) * w


# ---------------------------------------------------------------------------
# Theorem 2.1: explicit coefficients (exactly as printed in the paper)
# ---------------------------------------------------------------------------
def coefficients(s, alpha, L):
    """Return the twelve coefficient functions (A1,...,F1, A2,...,F2) of Theorem 2.1."""
    s = np.asarray(s, dtype=float)
    a = alpha
    den = 2.0 * L**2 * a**5 * np.sinh(a * L)
    A1 = -(L**2 - 2 * L * s + s**2) * s / (a**4 * L**2)
    B1 = (L**3 - 3 * L * s**2 + 2 * s**3) / (a**4 * L**3)
    C1 = -(L**2 - 2 * L * s + s**2) * s / (2 * a**2 * L**2)
    D1 = (L**3 - 3 * L * s**2 + 2 * s**3) / (6 * a**2 * L**3)
    E1 = -(a * s * (L - s) * ((L - s) * np.exp(-a * L) - s) + L**2 * np.sinh(a * (L - s))) / den
    F1 = (a * s * (L - s) * ((L - s) * np.exp(a * L) - s) + L**2 * np.sinh(a * (L - s))) / den
    A2 = (a**2 * L**2 * s + 12 * L - 6 * s) * s**2 / (6 * a**4 * L**2)
    B2 = -(a**2 * L**3 + 6 * L - 4 * s) * s**2 / (2 * a**4 * L**3)
    C2 = (2 * L - s) * s**2 / (2 * a**2 * L**2)
    D2 = -(3 * L - 2 * s) * s**2 / (6 * a**2 * L**3)
    E2 = -(a * s * (L - s) * ((L - s) * np.exp(-a * L) - s) - L**2 * np.exp(-a * L) * np.sinh(a * s)) / den
    F2 = (a * s * (L - s) * ((L - s) * np.exp(a * L) - s) - L**2 * np.exp(a * L) * np.sinh(a * s)) / den
    return (A1, B1, C1, D1, E1, F1), (A2, B2, C2, D2, E2, F2)


def G0_raw(x, s, alpha=1.0, L=1.0):
    """Explicit Green's function G0(x,s) of Theorem 2.1 (non-positive).

    This raw coefficient expression is retained for algebra checks. Use
    -Gt for numerical evaluations: Gt selects positive quadrature at small
    alpha*L and an overflow-free form otherwise.
    """
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    (A1, B1, C1, D1, E1, F1), (A2, B2, C2, D2, E2, F2) = coefficients(s, alpha, L)
    ea, em = np.exp(alpha * x), np.exp(-alpha * x)
    left = A1 + B1 * x + C1 * x**2 + D1 * x**3 + E1 * ea + F1 * em
    right = A2 + B2 * x + C2 * x**2 + D2 * x**3 + E2 * ea + F2 * em
    return np.where(x <= s, left, right)


def Gt(x, s, alpha=1.0, L=1.0):
    """Positive response kernel with exact endpoint values.

    For alpha*L < 0.8, evaluate the positive factorized integral to avoid
    catastrophic cancellation. Otherwise use the overflow-free closed form.
    The coefficient function G0_raw is retained independently for algebra checks.
    """
    x, s = np.broadcast_arrays(np.asarray(x, float), np.asarray(s, float))
    if alpha * L < 0.8:
        out = Gt_integral(x, s, alpha, L)
    else:
        out = Gt_stable(x, s, alpha, L)
    return np.where((x == 0) | (x == L) | (s == 0) | (s == L), 0.0, out)


def G0(x, s, alpha=1.0, L=1.0):
    """Operational nonpositive kernel, stable at small and large alpha*L."""
    return -Gt(x, s, alpha, L)


def G2(x, xi, alpha=1.0, L=1.0):
    """Overflow-free Dirichlet Green's function of v'' - alpha^2 v."""
    x, xi = np.broadcast_arrays(np.asarray(x, float), np.asarray(xi, float))
    lo, hi = np.minimum(x, xi), np.maximum(x, xi)
    H = (np.exp(alpha * (lo - hi)) * np.expm1(-2 * alpha * lo)
         * np.expm1(-2 * alpha * (L - hi))
         / (-np.expm1(-2 * alpha * L)) / (2 * alpha))
    return -H


def Gt_integral(x, s, alpha=1.0, L=1.0, n=32):
    """Vectorized positive convolution, split at both x and s.

    Used for small alpha*L and for near-endpoint cone optimization, where
    cancellation in the closed form can otherwise corrupt small values.
    """
    x, s = np.broadcast_arrays(np.asarray(x, float), np.asarray(s, float))
    t, w = gauss_legendre(n, 0, 1)
    cuts = [np.zeros_like(x), np.minimum(x, s), np.maximum(x, s), np.full_like(x, L)]
    out = np.zeros_like(x)
    for a, b in zip(cuts[:-1], cuts[1:]):
        q = a[..., None] + (b - a)[..., None] * t
        out += (b-a) * np.sum(w * (-G2(x[..., None], q, alpha, L)) * G4(q, s[..., None], L), axis=-1)
    return out


def G4(x, s, L=1.0):
    """Clamped Euler--Bernoulli kernel: w'''' = delta(x-s), w = w' = 0 at 0, L (non-negative)."""
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    left = x**2 * (L - s) ** 2 * (3 * s * L - 2 * s * x - L * x)
    right = s**2 * (L - x) ** 2 * (3 * x * L - 2 * s * x - L * s)
    return np.where(x <= s, left, right) / (6 * L**3)


def G4_xx(x, s, L=1.0):
    """Second x-derivative of G4 (bending-moment influence function of the clamped beam)."""
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    # left piece (L-s)^2 [3sL x^2 - (2s+L) x^3]
    left = (L - s) ** 2 * (6 * s * L - 6 * (2 * s + L) * x)
    # right piece s^2 p(x), p = (L-x)^2 (c x - L s), c = 3L - 2s, p'' = 2(cx - Ls) - 4c(L-x)
    c = 3 * L - 2 * s
    right = s**2 * (2 * (c * x - L * s) - 4 * c * (L - x))
    return np.where(x <= s, left, right) / (6 * L**3)


def Gt_stable(x, s, alpha=1.0, L=1.0):
    r"""Overflow-free form of Gtilde; use the convolution for small alpha*L:

    Gt = G4/alpha^2 + [d_xx G4 - k0(s) S(L-x) - kL(s) S(x)]/alpha^4 + (-G2)/alpha^4,

    with S(y) = sinh(alpha y)/sinh(alpha L), k0 = d_xx G4(0,s), kL = d_xx G4(L,s).
    This is the operational closed form for alpha*L >= 0.8. Positive
    convolution is used for smaller alpha*L.
    """
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))

    def S(y):
        y = np.asarray(y, dtype=float)
        return np.exp(alpha * (y - L)) * (-np.expm1(-2 * alpha * y)) / (-np.expm1(-2 * alpha * L))

    lo, hi = np.minimum(x, s), np.maximum(x, s)
    mG2 = (np.exp(alpha * (lo - hi)) * np.expm1(-2 * alpha * lo) * np.expm1(-2 * alpha * (L - hi))
           / (-np.expm1(-2 * alpha * L)) / (2 * alpha))
    k0 = G4_xx(0.0 * s, s, L)
    kL = G4_xx(L + 0.0 * s, s, L)
    return (G4(x, s, L) / alpha**2
            + (G4_xx(x, s, L) - k0 * S(L - x) - kL * S(x)) / alpha**4
            + mG2 / alpha**4)


def Gt_conv(x, s, alpha=1.0, L=1.0, n=60):
    """Factorised representation Gt(x,s) = int_0^L (-G2(x,xi)) G4(xi,s) dxi (Lemma 2.1).

    The integrand has kinks at xi = x and xi = s, so the interval is split there.
    """
    x = float(x)
    s = float(s)
    cuts = sorted({0.0, min(x, s), max(x, s), L})
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a <= 0:
            continue
        xi, w = gauss_legendre(n, a, b)
        total += np.sum(w * (-G2(x, xi, alpha, L)) * G4(xi, s, L))
    return total


# ---------------------------------------------------------------------------
# Uniform-load response and the constants A, B
# ---------------------------------------------------------------------------
def sech(z):
    z = np.abs(np.asarray(z, dtype=float))
    return 2.0 * np.exp(-z) / (1.0 + np.exp(-2.0 * z))


def uniform_load_response(x, alpha=1.0, L=1.0):
    r"""u1(x) = int_0^L Gt(x,s) ds, the solution for f = 1 (closed form)."""
    x = np.asarray(x, dtype=float)
    if alpha * L < 0.8:
        t, w = gauss_legendre(40, 0, 1)
        out = np.zeros_like(x)
        for a, b in ((np.zeros_like(x), x), (x, np.full_like(x, L))):
            q = a[..., None] + (b-a)[..., None] * t
            out += (b-a) * np.sum(w * (-G2(x[..., None], q, alpha, L)) * q**2 * (L-q)**2 / 24, axis=-1)
        return out
    P = x**2 * (L - x) ** 2 / 24.0
    P2 = (6 * x**2 - 6 * L * x + L**2) / 12.0
    c = L**2 / (12 * alpha**4) + 1.0 / alpha**6
    # cosh(alpha (x - L/2)) / cosh(alpha L/2), overflow free
    z = alpha * np.abs(x - L / 2)
    h = alpha * L / 2
    ratio = np.exp(z - h) * (1 + np.exp(-2 * z)) / (1 + np.exp(-2 * h))
    return P / alpha**2 + P2 / alpha**4 + 1.0 / alpha**6 - c * ratio


def size_effect_factor(kappa):
    r"""D(kappa) = u1(L/2) / (L^4/(384 alpha^2)), kappa = alpha L = L/g.

    D(kappa) = 1 - 16/kappa^2 + 384/kappa^4 - (32/kappa^2 + 384/kappa^4) sech(kappa/2).
    For small kappa a Taylor series is used to avoid cancellation.
    """
    k = np.atleast_1d(np.asarray(kappa, dtype=float))
    out = np.empty_like(k)
    big = k >= 0.6
    kb = k[big]
    out[big] = 1 - 16 / kb**2 + 384 / kb**4 - (32 / kb**2 + 384 / kb**4) * sech(kb / 2)
    ks = k[~big]
    if ks.size:
        # exact series from mpmath (see scripts/compute_results.py)
        coef = _D_SERIES
        z = ks**2
        out[~big] = z * np.polyval(coef[::-1], z)
    return out if np.ndim(kappa) else float(out[0])


# Taylor coefficients of D(kappa)/kappa^2 in powers of kappa^2 (computed symbolically)
_D_SERIES = None


def _init_series():
    global _D_SERIES
    import sympy as sp
    k = sp.symbols('k')
    D = 1 - 16 / k**2 + 384 / k**4 - (32 / k**2 + 384 / k**4) * sp.sech(k / 2)
    ser = sp.series(D / k**2, k, 0, 18).removeO()
    _D_SERIES = np.array([float(ser.coeff(k, 2 * j)) for j in range(9)])


_init_series()


def A_const(alpha=1.0, L=1.0):
    """A = max_x int_0^L Gt(x,s) ds = u1(L/2) (closed form)."""
    import mpmath as mp
    with mp.workdps(60):
        a, ell = mp.mpf(str(alpha)), mp.mpf(str(L))
        k = a * ell
        D = 1 - 16/k**2 + 384/k**4 - (32/k**2+384/k**4)/mp.cosh(k/2)
        return float(ell**4 * D / (384*a**2))


def int_Gt_over(x, a, b, alpha=1.0, L=1.0, n=40):
    """int_a^b Gt(x,s) ds by Gauss--Legendre quadrature (split at s = x)."""
    x = float(x)
    cuts = [a, b] if not (a < x < b) else [a, x, b]
    total = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        s, w = gauss_legendre(n, lo, hi)
        total += np.sum(w * Gt(x, s, alpha, L))
    return total


def B_const(delta=0.25, alpha=1.0, L=1.0):
    """B = min_{x in [delta, L-delta]} int_delta^{L-delta} Gt(x,s) ds."""
    f = lambda x: int_Gt_over(x, delta, L - delta, alpha, L)
    xs = np.linspace(delta, L - delta, 201)
    vals = np.array([f(x) for x in xs])
    i = int(np.argmin(vals))
    best = vals[i]
    # refine (the minimum is attained at the end points by symmetry/unimodality)
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    if hi > lo:
        res = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
        best = min(best, res.fun)
    return float(best)


def _max_in_x(s, alpha, L, ny=401):
    ys = np.linspace(0, L, ny)
    v = Gt_integral(ys, s, alpha, L)
    j = int(np.argmax(v))
    lo, hi = ys[max(j - 1, 0)], ys[min(j + 1, ny - 1)]
    res = minimize_scalar(lambda y: -float(Gt_integral(y, s, alpha, L)), bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-13})
    return max(float(v[j]), -res.fun)


def _min_in_x(s, delta, alpha, L, nx=201):
    xs = np.linspace(delta, L - delta, nx)
    v = Gt_integral(xs, s, alpha, L)
    j = int(np.argmin(v))
    lo, hi = xs[max(j - 1, 0)], xs[min(j + 1, nx - 1)]
    res = minimize_scalar(lambda y: float(Gt_integral(y, s, alpha, L)), bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-13})
    return min(float(v[j]), res.fun)


def eta_ratio(s, delta=0.25, alpha=1.0, L=1.0):
    """r(s) = min_{x in [delta,L-delta]} Gt(x,s) / max_{y in [0,L]} Gt(y,s), 0 < s < L."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.array([_min_in_x(si, delta, alpha, L) / _max_in_x(si, alpha, L) for si in s])
    return out


def eta_endpoint_limit(delta=0.25, alpha=1.0, L=1.0, n=80):
    r"""r(0+) = min H0 / max H0 with H0(x) = int (-G2(x,xi)) K0(xi) dxi, K0 = xi (L-xi)^2/(2L^2)."""
    def H0(x):
        cuts = [0.0, x, L]
        tot = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a <= 0:
                continue
            xi, w = gauss_legendre(n, a, b)
            tot += np.sum(w * (-G2(x, xi, alpha, L)) * xi * (L - xi) ** 2 / (2 * L**2))
        return tot
    xs = np.linspace(delta, L - delta, 401)
    mn = min(H0(x) for x in xs)
    ys = np.linspace(0, L, 801)
    vals = np.array([H0(y) for y in ys])
    j = int(np.argmax(vals))
    res = minimize_scalar(lambda y: -H0(y), bounds=(ys[max(j - 1, 0)], ys[min(j + 1, 800)]),
                          method='bounded', options={'xatol': 1e-13})
    mx = max(vals[j], -res.fun)
    return mn / mx


def eta_star(delta=0.25, alpha=1.0, L=1.0, ns=400):
    """Numerical estimate of eta*: sampled optimization including endpoint limits.

    This is not a certified global lower bound. Use eta_helmholtz for an
    analytically guaranteed cone constant in existence conditions.
    """
    s = np.linspace(L * 1e-3, L * (1 - 1e-3), ns)
    r = eta_ratio(s, delta, alpha, L)
    i = int(np.argmin(r))
    best = float(r[i])
    lo, hi = s[max(i - 1, 0)], s[min(i + 1, ns - 1)]
    res = minimize_scalar(lambda t: float(eta_ratio(t, delta, alpha, L)[0]), bounds=(lo, hi),
                          method='bounded', options={'xatol': 1e-10})
    best = min(best, res.fun)
    # the infimum may be approached at s -> 0+ (and, by symmetry, s -> L-)
    best = min(best, eta_endpoint_limit(delta, alpha, L))
    return float(best)


def eta_helmholtz(delta=0.25, alpha=1.0, L=1.0):
    """Analytical lower cone constant sinh(alpha delta)/sinh(alpha L)."""
    return float(np.exp(alpha*(delta-L)) * (-np.expm1(-2*alpha*delta)) / (-np.expm1(-2*alpha*L)))
