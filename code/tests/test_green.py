"""Tests of the Green's function (Theorem 2.1, Lemmas 2.1-2.3) and of the constants."""
import numpy as np
import pytest

from sgbeam import green as gr


@pytest.mark.parametrize("alpha,L", [(1.0, 1.0), (0.5, 2.0), (3.0, 1.5), (10.0, 1.0)])
def test_explicit_formula_equals_factorised_form(alpha, L):
    """G0 = int G2 G4 (Lemma 2.1) and the overflow-free form agree with Theorem 2.1."""
    rng = np.random.default_rng(1)
    for x, s in rng.uniform(0.02, 0.98, size=(10, 2)) * L:
        g = gr.Gt(x, s, alpha, L)
        assert abs(g + gr.G0_raw(x, s, alpha, L)) <= 1e-7 * g
        assert abs(g - gr.Gt_conv(x, s, alpha, L, n=80)) <= 1e-7 * g
        assert abs(g - gr.Gt_stable(x, s, alpha, L)) <= 1e-7 * g


def test_sign_of_G0():
    """G0 <= 0 on [0,L]^2 and G0 < 0 inside (Lemma 2.1)."""
    xs = np.linspace(0, 1, 101)
    X, S = np.meshgrid(xs, xs, indexing="ij")
    for alpha in (0.3, 1.0, 5.0, 20.0):
        G = gr.G0(X, S, alpha, 1.0)                      # operational stable kernel
        assert G.max() <= 1e-8 * abs(G).max()
        Gs = -gr.Gt_stable(X, S, alpha, 1.0)             # overflow-free form
        assert np.all(Gs[1:-1, 1:-1] < 0)


def test_G4_positive_and_normalised():
    xs = np.linspace(0, 2, 81)
    X, S = np.meshgrid(xs, xs, indexing="ij")
    assert np.all(gr.G4(X, S, 2.0) >= -1e-15)
    # w'''' = 1 with clamped ends: int G4(x,s) ds = x^2 (L-x)^2 / 24
    s, w = gr.gauss_legendre(60, 0, 2.0)
    for x in (0.3, 1.0, 1.7):
        val = np.sum(w * gr.G4(x, s, 2.0) * (s < x)) + 0.0
        s1, w1 = gr.gauss_legendre(40, 0, x)
        s2, w2 = gr.gauss_legendre(40, x, 2.0)
        val = np.sum(w1 * gr.G4(x, s1, 2.0)) + np.sum(w2 * gr.G4(x, s2, 2.0))
        assert np.isclose(val, x**2 * (2 - x) ** 2 / 24, rtol=1e-12)


def test_constants_of_example_4_1():
    assert np.isclose(gr.A_const(1.0), 2.170423657532656e-4, rtol=1e-11)
    assert np.isclose(gr.B_const(0.25, 1.0), 1.181999127729831e-4, rtol=1e-10)
    eta = gr.eta_star(0.25, 1.0)
    assert np.isclose(eta, 0.601350, atol=2e-6)
    assert gr.eta_helmholtz(0.25, 1.0) < eta


def test_uniform_load_response_and_size_law():
    xs = np.linspace(0, 1, 201)
    for alpha in (0.7, 1.0, 5.0):
        num = np.array([gr.int_Gt_over(x, 0.0, 1.0, alpha, 1.0, n=50) for x in xs])
        assert np.allclose(num, gr.uniform_load_response(xs, alpha), rtol=1e-8, atol=1e-16)
        assert np.isclose(gr.A_const(alpha), gr.size_effect_factor(alpha) / (384 * alpha**2), rtol=1e-12)
    k = np.geomspace(1e-2, 1e3, 500)
    D = gr.size_effect_factor(k)
    assert np.all(np.diff(D) > 0) and np.all((D > 0) & (D < 1))
    assert np.isclose(1e3**2 * (1 - gr.size_effect_factor(1e3)), 16.0, rtol=1e-3)
    assert np.isclose(gr.size_effect_factor(1e-2) / 1e-4, 11 / 120, rtol=1e-4)


def test_small_parameter_and_large_parameter_kernels():
    """The operational kernel remains accurate where the coefficient form is ill-conditioned."""
    for alpha, L in [(1e-4, 1.0), (0.1, 2.0), (500.0, 1.0)]:
        for x,s in [(0.13*L,0.37*L), (0.71*L,0.91*L)]:
            actual=gr.Gt(x,s,alpha,L)
            reference=gr.Gt_conv(x,s,alpha,L,n=200)
            assert actual>0
            assert np.isclose(actual,reference,rtol=1e-10,atol=1e-22)
        assert gr.Gt(0,0.5*L,alpha,L)==0
        assert gr.Gt(0.5*L,L,alpha,L)==0


def test_explicit_cone_bound_for_noncentral_sources():
    for alpha,L,delta in [(1.0,1.0,0.25),(5.0,1.0,0.25),(2.0,3.0,0.8)]:
        gamma=gr.eta_helmholtz(delta,alpha,L)
        x=np.linspace(delta,L-delta,75)
        y=np.linspace(0,L,151)
        for s in [0.001*L,0.27*L,0.7*L,0.999*L]:
            ratio=np.min(gr.Gt_integral(x,s,alpha,L))/np.max(gr.Gt_integral(y,s,alpha,L))
            assert ratio>=gamma
