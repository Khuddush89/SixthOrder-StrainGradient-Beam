#!/usr/bin/env python3
"""
Compute every number quoted in the paper and store it in results/results.json.

    python scripts/compute_results.py          (about one minute)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from sgbeam import beams, green as gr, solvers as so  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "results.json")
EXP = (lambda x, u: np.exp(u), lambda x, u: np.exp(u))
ONE = (lambda x, u: np.ones_like(np.asarray(u, float)), lambda x, u: np.zeros_like(np.asarray(u, float)))


def max_of(fun, lo=0.4, hi=0.6):
    res = minimize_scalar(lambda t: -fun(t), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return -res.fun, res.x


# ---------------------------------------------------------------------------
def sign_check():
    """Remark 2.2: max of G0 over grids for several alpha and L (should be <= 0)."""
    worst = -np.inf
    for alpha in np.linspace(0.2, 5.0, 20):
        for L in np.linspace(0.5, 3.0, 20):
            xs = np.linspace(0, L, 81)
            X, S = np.meshgrid(xs, xs, indexing="ij")
            worst = max(worst, float(np.max(-gr.Gt(X, S, alpha, L))))
    assert np.isfinite(worst) and worst <= 0.0
    return {"max_G0": worst, "method": "operational stable kernel; sampled check"}


def constants():
    """Constants of Example 4.1 (L = alpha = 1, delta = 1/4) and of Figure 3."""
    out = {"A": gr.A_const(1.0), "B": gr.B_const(0.25, 1.0), "eta_star": gr.eta_star(0.25, 1.0),
           "r_endpoint": gr.eta_endpoint_limit(0.25, 1.0)}
    # Diagnostic checks of the endpoint-sensitive numerical optimization
    out["eta_analytical"] = gr.eta_helmholtz(0.25, 1.0)
    out["r_at_s_0.001"] = float(gr.eta_ratio(1e-3, 0.25, 1.0)[0])
    # eta* as a function of alpha for three delta (Figure 3b)
    alphas = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    out["eta_vs_alpha"] = {"alpha": alphas,
                           **{f"delta={d}": [gr.eta_star(d, a) for a in alphas] for d in (0.125, 0.25, 0.375)}}
    return out


def example1():
    """Example 4.1: -u^(6) + u^(4) = e^u, L = alpha = 1."""
    gn = so.GreenNystrom(1.0, 1.0, n=48, m=40)
    f, fu = EXP
    um = gn.solve(f, fu, 1.0)
    picard_u, inc = gn.picard(f, 1.0)
    amps = np.concatenate([np.linspace(1e-4, 1.0, 60), np.linspace(1.02, 14.0, 400)])
    A, M, U = so.amplitude_branch(gn, f, fu, amps, mu0=0.46)
    k = int(np.argmax(M))
    uf, muf = so.fold_point(gn, f, fu, fu, U[k], M[k])
    j = k + int(np.where(M[k:] < 1.0)[0][0])
    up = gn.solve(f, fu, 1.0, u0=U[j])
    norm_p, _ = max_of(lambda t: gn.evaluate([t], f, up)[0])
    norm_m, _ = max_of(lambda t: gn.evaluate([t], f, um)[0])
    fold_amp, _ = max_of(lambda t: gn.evaluate([t], f, uf, muf)[0])
    # cross-checks: finite differences and finite elements
    xs = np.linspace(0, 1, 2001)
    up_c = gn.evaluate(xs, f, up)
    um_c = gn.evaluate(xs, f, um)
    dup = np.gradient(up_c, xs)
    fd = {}
    for N in (50, 100, 200):
        x, u = so.solve_fd(f, fu, 1.0, 1.0, N=N, u0=lambda x: np.interp(x, xs, up_c))
        x2, u2 = so.solve_fd(f, fu, 1.0, 1.0, N=N)
        fd[N] = {"plus": float(u.max()), "minus": float(u2.max())}
    fe = {}
    for ne in (16, 32, 64):
        fem = so.MixedFEM(1.0, 1.0, ne=ne)
        fem.solve(f, fu, 1.0, U0=fem.nodal_initial_guess(lambda x: np.interp(x, xs, up_c),
                                                          lambda x: np.interp(x, xs, dup)))
        p = float(fem.evaluate(xs).max())
        fem2 = so.MixedFEM(1.0, 1.0, ne=ne)
        fem2.solve(f, fu, 1.0)
        m = float(fem2.evaluate(xs).max())
        fe[ne] = {"plus": p, "minus": m}
    # cone property of the two solutions (delta = 1/4)
    mid = (xs >= 0.25) & (xs <= 0.75)
    return {"norm_minus": norm_m, "norm_plus": norm_p, "fold_mu": muf, "fold_amp": fold_amp,
            "u_minus_nodes": um.tolist(), "u_plus_nodes": up.tolist(),
            "picard_increments": [float(v) for v in inc[:6]],
            "cone_ratio_plus": float(up_c[mid].min() / up_c.max()),
            "cone_ratio_minus": float(um_c[mid].min() / um_c.max()),
            "fd": fd, "fe": fe,
            "branch": {"amp": A.tolist(), "mu": M.tolist()}}


def example2():
    """Example 4.2: load mu e^u for several alpha; admissible intensities."""
    f, fu = EXP
    rows = []
    branches = {}
    for alpha in (1.0, 2.0, 5.0, 10.0, 20.0):
        A = gr.A_const(alpha)
        B = gr.B_const(0.25, alpha)
        es = gr.eta_star(0.25, alpha)
        gn = so.GreenNystrom(alpha, 1.0, n=48, m=40)
        amps = np.concatenate([np.geomspace(5e-5, 1.0, 70), np.linspace(1.02, 12.0, 220)])
        Aa, M, U = so.amplitude_branch(gn, f, fu, amps, mu0=5e-5 / A)
        k = int(np.argmax(M))
        uf, muf = so.fold_point(gn, f, fu, fu, U[k], M[k])
        muK = 1.0 / (np.e * A)
        rows.append({"alpha": alpha, "A": A, "B": B, "eta_star": es, "mu_K": muK, "mu_fold": muf,
                     "ratio": muK / muf, "p_K": muK / alpha**2, "p_fold": muf / alpha**2,
                     "D": float(gr.size_effect_factor(alpha))})
        branches[str(alpha)] = {"amp": Aa.tolist(), "mu": M.tolist()}
    return {"rows": rows, "branches": branches, "p_classical": 384 / np.e}


def example3():
    """Example 4.3: f = lam (1 + x) u^2, alpha = 5, lam = 1e4."""
    alpha, lam, eta = 5.0, 1.0e4, 0.02
    f = lambda x, u: lam * (1 + x) * u**2
    fu = lambda x, u: 2 * lam * (1 + x) * u
    A, B, es = gr.A_const(alpha), gr.B_const(0.25, alpha), gr.eta_star(0.25, alpha)
    r = 1.0 / (lam * 2.0 * A)
    R = 1.0 / (lam * 1.25 * eta**2 * B)
    gn = so.GreenNystrom(alpha, 1.0, n=48, m=40)
    shape = gr.uniform_load_response(gn.nodes, alpha)
    u = gn.solve(f, fu, 1.0, u0=1.4 * shape / shape.max())
    nrm, xmax = max_of(lambda t: gn.evaluate([t], f, u)[0])
    xs = np.linspace(0, 1, 2001)
    uc = gn.evaluate(xs, f, u)
    fd, fe = {}, {}
    for N in (50, 100, 200):
        x, ufd = so.solve_fd(f, fu, alpha, 1.0, N=N, u0=lambda x: np.interp(x, xs, uc))
        fd[N] = float(ufd.max())
    du = np.gradient(uc, xs)
    for ne in (16, 32, 64):
        fem = so.MixedFEM(alpha, 1.0, ne=ne)
        fem.solve(f, fu, 1.0, U0=fem.nodal_initial_guess(lambda x: np.interp(x, xs, uc), lambda x: np.interp(x, xs, du)))
        fe[ne] = float(fem.evaluate(xs).max())
    mid = (xs >= 0.25) & (xs <= 0.75)
    return {"alpha": alpha, "lam": lam, "eta": eta, "A": A, "B": B, "eta_star": es, "r": 0.72, "R": 5600.0, "r_max": r, "R_min": R,
            "norm": nrm, "argmax": xmax, "u_nodes": u.tolist(),
            "fd": fd, "fe": fe,
            "cone_ratio": float(uc[mid].min() / uc.max())}


def verification():
    """Section 5: errors for f = 1 (exact solution u1) and alpha = 1."""
    out = {}
    for alpha in (1.0, 10.0):
        xs = np.linspace(0, 1, 2001)
        ex = gr.uniform_load_response(xs, alpha)
        scale = ex.max()
        fd = {}
        for N in (25, 50, 100, 200):
            x, u = so.solve_fd(*ONE, alpha, 1.0, N=N)
            fd[N] = float(np.max(np.abs(u - gr.uniform_load_response(x, alpha))) / scale)
        fe = {}
        for ne in (4, 8, 16, 32, 64):
            fem = so.MixedFEM(alpha, 1.0, ne=ne)
            fem.solve(*ONE)
            fe[ne] = float(np.max(np.abs(fem.evaluate(xs) - ex)) / scale)
        gn_err = {}
        for n in (4, 8, 16):
            gn = so.GreenNystrom(alpha, 1.0, n=n, m=40)
            u = gn.solve(*ONE)
            gn_err[n] = float(np.max(np.abs(gn.evaluate(xs, ONE[0], u) - ex)) / scale)
        out[str(alpha)] = {"fd": fd, "fe": fe, "gn": gn_err}
    return out


def mechanics():
    """Section 6: size-effect factor and comparison with shear-deformable beams."""
    nu = 0.3
    kc = beams.k_cowper(nu)
    kappa_list = [1, 2, 5, 10, 20, 50, 100]
    D = {str(k): float(gr.size_effect_factor(k)) for k in kappa_list}
    table = []
    for Lh in (5, 10, 20):
        row = {"L/h": Lh, "shear_cowper": float(beams.shear_ratio(Lh, nu, kc)),
               "shear_5_6": float(beams.shear_ratio(Lh, nu, beams.K_CONVENTIONAL))}
        for hg in (1, 2, 5, 10):
            row[f"sg_h/g={hg}"] = float(gr.size_effect_factor(Lh * hg))
        table.append(row)
    return {"nu": nu, "k_cowper": kc, "D": D, "table": table,
            "balance_shear_5_6": beams.balance_length(nu, beams.K_CONVENTIONAL),
            "balance_cowper": beams.balance_length(nu, kc),
            "D_small_coeff": float(gr._D_SERIES[0])}


def main():
    t0 = time.time()
    res = {}
    for name, fun in (("sign_check", sign_check), ("constants", constants), ("example1", example1),
                      ("example2", example2), ("example3", example3), ("verification", verification),
                      ("mechanics", mechanics)):
        t = time.time()
        res[name] = fun()
        print(f"{name:14s} done in {time.time() - t:5.1f} s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {OUT} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
