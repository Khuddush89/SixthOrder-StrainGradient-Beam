"""Tests of the three numerical methods (Section 5) and of the examples (Section 4)."""
import numpy as np

from sgbeam import beams, green as gr, solvers as so

ONE = (lambda x, u: np.ones_like(u), lambda x, u: np.zeros_like(u))
EXP = (lambda x, u: np.exp(u), lambda x, u: np.exp(u))


def test_orders_of_convergence():
    xs = np.linspace(0, 1, 801)
    ex = gr.uniform_load_response(xs, 1.0)
    e = [np.max(np.abs(so.solve_fd(*ONE, 1.0, 1.0, N=N)[1] - gr.uniform_load_response(np.linspace(0, 1, N + 1), 1.0)))
         for N in (25, 50, 100)]
    assert 1.9 < np.log2(e[0] / e[1]) < 2.1 and 1.9 < np.log2(e[1] / e[2]) < 2.1
    ef = []
    for ne in (8, 16, 32):
        fem = so.MixedFEM(1.0, 1.0, ne=ne)
        fem.solve(*ONE)
        ef.append(np.max(np.abs(fem.evaluate(xs) - ex)))
    assert np.log2(ef[0] / ef[1]) > 3.8 and np.log2(ef[1] / ef[2]) > 3.8
    gn = so.GreenNystrom(1.0, 1.0, n=12)
    u = gn.solve(*ONE)
    assert np.max(np.abs(gn.evaluate(xs, ONE[0], u) - ex)) < 1e-10 * ex.max()


def test_example_4_1_two_solutions():
    gn = so.GreenNystrom(1.0, 1.0, n=40)
    f, fu = EXP
    um = gn.solve(f, fu, 1.0)
    assert np.isclose(gn.evaluate([0.5], f, um)[0], 2.1708137e-4, rtol=1e-6)
    amps = np.concatenate([np.linspace(1e-4, 1.0, 40), np.linspace(1.05, 12.5, 200)])
    A, M, U = so.amplitude_branch(gn, f, fu, amps, mu0=0.46)
    k = int(np.argmax(M))
    assert np.isclose(M[k], 2004.1, rtol=1e-3)
    j = k + int(np.where(M[k:] < 1.0)[0][0])
    up = gn.solve(f, fu, 1.0, u0=U[j])
    norm = gn.evaluate([0.5], f, up)[0]
    assert np.isclose(norm, 11.840388, rtol=1e-6)
    # finite elements reproduce the large solution
    xs = np.linspace(0, 1, 1001)
    v = gn.evaluate(xs, f, up)
    fem = so.MixedFEM(1.0, 1.0, ne=32)
    fem.solve(f, fu, 1.0, U0=fem.nodal_initial_guess(lambda x: np.interp(x, xs, v),
                                                      lambda x: np.interp(x, xs, np.gradient(v, xs))))
    assert np.isclose(fem.evaluate([0.5])[0], norm, rtol=1e-6)


def test_example_4_3():
    lam, alpha = 1e4, 5.0
    f = lambda x, u: lam * (1 + x) * u**2
    fu = lambda x, u: 2 * lam * (1 + x) * u
    gn = so.GreenNystrom(alpha, 1.0, n=40)
    sh = gr.uniform_load_response(gn.nodes, alpha)
    u = gn.solve(f, fu, 1.0, u0=1.4 * sh / sh.max())
    xs = np.linspace(0, 1, 2001)
    v = gn.evaluate(xs, f, u)
    r = 1 / (2 * lam * gr.A_const(alpha))
    R = 4 / (5 * lam * 0.02**2 * gr.B_const(0.25, alpha))
    assert r <= v.max() <= R
    assert np.isclose(v.max(), 1.38165, atol=2e-5)


def test_shear_formulas():
    assert np.isclose(beams.k_cowper(0.3), 0.8497, atol=1e-4)
    assert np.isclose(beams.shear_ratio(10, 0.3, 5 / 6), 1.1248)
    assert np.isclose(beams.balance_length(0.3, 5 / 6), 0.883, atol=1e-3)
