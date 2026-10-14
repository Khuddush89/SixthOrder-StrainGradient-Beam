r"""
Numerical solvers for the nonlinear sixth-order problem

    -u^(6) + alpha^2 u^(4) = f(x, u),   0 < x < L,
    u = u'' = 0,  u''' - alpha^2 u' = 0   at x = 0 and x = L.

Three independent discretisations:

(M1) ``GreenNystrom``  Nystrom (product-integration) method for the Hammerstein
     equation u = T u, (T u)(x) = int Gt(x,s) f(s,u(s)) ds, with the explicit
     Green's function of Theorem 2.1; spectral accuracy.
(M2) ``solve_fd``      second-order central finite differences applied directly
     to the sixth-order equation; the boundary conditions are imposed with two
     ghost nodes at each end.  Uses neither the Green's function nor the
     factorisation.
(M3) ``MixedFEM``      finite elements for the factorised system of Lemma 2.1,
         w = u'' - alpha^2 u,   -w'''' = f(x,u),   w = w' = 0 at 0, L,
         u'' - alpha^2 u = w,    u = 0 at 0, L,
     with C^1 cubic Hermite elements for both fields.

All nonlinear systems are solved by Newton's method.  ``amplitude_branch``
traces the solution set of u = mu T e^u (or any load mu f) parametrised by the
mid-span amplitude, which passes through turning points.

Author: M. Khuddush.  Licence: MIT.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from . import green as gr

Fun = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ===========================================================================
# (M1) Green--Nystrom method with product integration
# ===========================================================================
def chebyshev_nodes(n: int, L: float = 1.0) -> np.ndarray:
    """Chebyshev points of the first kind mapped to (0, L), increasing."""
    k = np.arange(n)
    return 0.5 * L * (1.0 - np.cos((2 * k + 1) * np.pi / (2 * n)))


def lagrange_matrix(nodes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Matrix of Lagrange basis values l_j(t_i) at Chebyshev points of the first
    kind (barycentric formula with the weights (-1)^k sin(theta_k))."""
    n = len(nodes)
    k = np.arange(n)
    w = (-1.0) ** k * np.sin((2 * k + 1) * np.pi / (2 * n))
    t = np.atleast_1d(t)
    diff = t[:, None] - nodes[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-15)
    diff[exact] = 1.0
    M = w[None, :] / diff
    M = M / M.sum(axis=1, keepdims=True)
    rows = np.where(exact.any(axis=1))[0]
    for i in rows:
        M[i] = 0.0
        M[i, np.argmax(exact[i])] = 1.0
    return M


class GreenNystrom:
    """u(x_i) = sum_j W_ij f(x_j, u_j),  W_ij = int Gt(x_i, s) l_j(s) ds."""

    def __init__(self, alpha: float, L: float = 1.0, n: int = 48, m: int = 40):
        self.alpha, self.L, self.n, self.m = alpha, L, n, m
        self.nodes = chebyshev_nodes(n, L)
        self.W = self.matrix(self.nodes)

    def matrix(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        W = np.zeros((len(x), self.n))
        for i, xi in enumerate(x):
            for a, b in ((0.0, xi), (xi, self.L)):
                if b - a <= 0:
                    continue
                q, w = gr.gauss_legendre(self.m, a, b)
                W[i] += (w * gr.Gt(xi, q, self.alpha, self.L)) @ lagrange_matrix(self.nodes, q)
        return W

    def solve(self, f: Fun, fu: Fun, mu: float = 1.0, u0=None, tol=1e-13, maxit=80):
        """Newton's method for u = mu W f(x, u); returns nodal values."""
        x = self.nodes
        u = np.zeros(self.n) if u0 is None else np.array(u0, dtype=float)
        for _ in range(maxit):
            F = u - mu * self.W @ f(x, u)
            J = np.eye(self.n) - mu * self.W * fu(x, u)[None, :]
            du = np.linalg.solve(J, -F)
            u = u + du
            if np.max(np.abs(du)) <= tol * max(1.0, np.max(np.abs(u))):
                return u
        raise RuntimeError("Green-Nystrom Newton iteration did not converge")

    def picard(self, f: Fun, mu: float = 1.0, tol=1e-15, maxit=200):
        """Picard iteration u_{k+1} = mu T f(u_k) from u_0 = 0 (returns u and the increments)."""
        x = self.nodes
        u = np.zeros(self.n)
        inc = []
        for _ in range(maxit):
            un = mu * self.W @ f(x, u)
            inc.append(np.max(np.abs(un - u)))
            u = un
            if inc[-1] <= tol * max(1.0, np.max(np.abs(u))):
                break
        return u, np.array(inc)

    def evaluate(self, x, f: Fun, u, mu: float = 1.0):
        """Continuous solution u(x) = mu int Gt(x, s) f(s, u(s)) ds from the nodal values."""
        return mu * self.matrix(x) @ f(self.nodes, u)


def amplitude_branch(gn: GreenNystrom, f: Fun, fu: Fun, amps, mu0: float, u0=None,
                     tol=1e-12, maxit=60):
    """Solutions of u = mu T f(u) parametrised by the mid-span value a = u(L/2).

    Unknowns (u, mu); equations u - mu W f(u) = 0 and mu w_mid . f(u) = a.
    Returns arrays (a, mu, U) for the amplitudes that converged.
    """
    x = gn.nodes
    n = len(x)
    wmid = gn.matrix([0.5 * gn.L])[0]
    mu = float(mu0)
    u = np.full(n, amps[0]) if u0 is None else np.array(u0, dtype=float)
    A, M, U = [], [], []
    for a in amps:
        ok = False
        for _ in range(maxit):
            fv, f1 = f(x, u), fu(x, u)
            R = np.concatenate([u - mu * gn.W @ fv, [mu * wmid @ fv - a]])
            J = np.zeros((n + 1, n + 1))
            J[:n, :n] = np.eye(n) - mu * gn.W * f1[None, :]
            J[:n, n] = -gn.W @ fv
            J[n, :n] = mu * wmid * f1
            J[n, n] = wmid @ fv
            d = np.linalg.solve(J, -R)
            u, mu = u + d[:n], mu + d[n]
            if np.max(np.abs(d[:n])) < tol * max(1.0, np.max(np.abs(u))) and abs(d[n]) < tol * max(1.0, abs(mu)):
                ok = True
                break
        if not ok:
            break
        A.append(a)
        M.append(mu)
        U.append(u.copy())
    return np.array(A), np.array(M), np.array(U)


def fold_point(gn: GreenNystrom, f: Fun, fu: Fun, fuu: Fun, u0, mu0, tol=1e-12, maxit=60):
    """Turning point of mu -> u(mu) (Moore--Spence extended system)."""
    x, W, n = gn.nodes, gn.W, gn.n
    u, mu = np.array(u0, dtype=float), float(mu0)
    J0 = np.eye(n) - mu * W * fu(x, u)[None, :]
    phi = np.linalg.svd(J0)[2][-1]
    if phi.sum() < 0:
        phi = -phi
    c = phi / (phi @ phi)
    for _ in range(maxit):
        fv, f1, f2 = f(x, u), fu(x, u), fuu(x, u)
        Fu = np.eye(n) - mu * W * f1[None, :]
        R = np.concatenate([u - mu * W @ fv, Fu @ phi, [c @ phi - 1.0]])
        J = np.zeros((2 * n + 1, 2 * n + 1))
        J[:n, :n] = Fu
        J[:n, 2 * n] = -W @ fv
        J[n:2 * n, :n] = -mu * W * (f2 * phi)[None, :]
        J[n:2 * n, n:2 * n] = Fu
        J[n:2 * n, 2 * n] = -W @ (f1 * phi)
        J[2 * n, n:2 * n] = c
        d = np.linalg.solve(J, -R)
        u, phi, mu = u + d[:n], phi + d[n:2 * n], mu + d[2 * n]
        if np.max(np.abs(d)) < tol * max(1.0, abs(mu)):
            return u, mu
    raise RuntimeError("fold_point did not converge")


# ===========================================================================
# (M2) finite differences for the sixth-order equation
# ===========================================================================
def fd_matrix(N: int, alpha: float, L: float = 1.0):
    """Sparse matrix acting on (u_{-2}, ..., u_{N+2}); rows 0..N-2 are the
    equation -D^6 u + alpha^2 D^4 u at x_1..x_{N-1}, the last six rows the
    boundary conditions (all second-order accurate)."""
    h = L / N
    n = N + 5
    off = 2
    rows, cols, vals = [], [], []
    c6 = np.array([1, -6, 15, -20, 15, -6, 1]) / h**6
    c4 = np.array([1, -4, 6, -4, 1]) / h**4
    for i in range(1, N):
        r = i - 1
        for k, c in enumerate(c6):
            rows.append(r); cols.append(off + i - 3 + k); vals.append(-c)
        for k, c in enumerate(c4):
            rows.append(r); cols.append(off + i - 2 + k); vals.append(alpha**2 * c)
    bc = []
    for i in (0, N):
        j = off + i
        bc.append({j: 1.0})                                                # u = 0
        bc.append({j - 1: 1 / h**2, j: -2 / h**2, j + 1: 1 / h**2})          # u'' = 0
        d = {j + 2: 1 / (2 * h**3), j + 1: -1 / h**3, j - 1: 1 / h**3, j - 2: -1 / (2 * h**3)}
        d[j + 1] += -alpha**2 / (2 * h)
        d[j - 1] += alpha**2 / (2 * h)
        bc.append(d)                                                       # u''' - a^2 u' = 0
    for k, d in enumerate(bc):
        for c, v in d.items():
            rows.append(N - 1 + k); cols.append(c); vals.append(v)
    return sps.csr_matrix((vals, (rows, cols)), shape=(n, n)), np.linspace(0, L, N + 1)


def solve_fd(f: Fun, fu: Fun, alpha: float, L: float = 1.0, N: int = 200, mu: float = 1.0,
             u0=None, tol=1e-11, maxit=60):
    """Newton's method for the finite-difference equations; returns (x, u)."""
    A, x = fd_matrix(N, alpha, L)
    n, off = N + 5, 2
    inner = np.arange(1, N)
    U = np.zeros(n)
    if u0 is not None:
        U[off:off + N + 1] = u0(x) if callable(u0) else np.interp(x, np.linspace(0, L, len(u0)), u0)
    prev = np.inf
    for it in range(maxit):
        ui = U[off + inner]
        F = A @ U
        F[:N - 1] -= mu * f(x[inner], ui)
        Jd = sps.csr_matrix((mu * fu(x[inner], ui), (np.arange(N - 1), off + inner)), shape=(n, n))
        dU = spla.spsolve((A - Jd).tocsc(), -F)
        U += dU
        step = np.max(np.abs(dU)) / max(1.0, np.max(np.abs(U)))
        if step <= tol:
            break
        if it >= 4 and step < 1e-6 and step > 0.5 * prev:     # rounding level reached
            break
        prev = step
    else:
        raise RuntimeError("finite-difference Newton iteration did not converge")
    return x, U[off:off + N + 1].copy()


# ===========================================================================
# (M3) mixed finite elements for the factorised system
# ===========================================================================
def hermite(t, h):
    """Cubic Hermite shape functions on an element of length h at t in [0,1],
    with first and second x-derivatives."""
    N = np.array([1 - 3 * t**2 + 2 * t**3, h * (t - 2 * t**2 + t**3), 3 * t**2 - 2 * t**3, h * (-t**2 + t**3)])
    dN = np.array([-6 * t + 6 * t**2, h * (1 - 4 * t + 3 * t**2), 6 * t - 6 * t**2, h * (-2 * t + 3 * t**2)]) / h
    d2N = np.array([-6 + 12 * t, h * (-4 + 6 * t), 6 - 12 * t, h * (-2 + 6 * t)]) / h**2
    return N, dN, d2N


class MixedFEM:
    r"""Weak form of the factorised system (Lemma 2.1):

        int w'' psi'' dx = - int f(x,u) psi dx      for psi in H^2_0,   w in H^2_0,
        int (u' phi' + alpha^2 u phi) dx = - int w phi dx   for phi in H^1_0,  u in H^1_0,

    discretised with cubic Hermite elements for w and u; Newton for the coupled system.
    """

    def __init__(self, alpha: float, L: float = 1.0, ne: int = 64, nq: int = 6):
        self.alpha, self.L, self.ne = alpha, L, ne
        self.h = h = L / ne
        self.xn = np.linspace(0, L, ne + 1)
        tq, wq = gr.gauss_legendre(nq, 0.0, 1.0)
        self.tq, self.wq = tq, wq
        self.N, self.dN, self.d2N = hermite(tq, h)
        nd = 2 * (ne + 1)
        K2 = np.zeros((nd, nd)); K1 = np.zeros((nd, nd)); M = np.zeros((nd, nd))
        for e in range(ne):
            I = np.arange(2 * e, 2 * e + 4)
            K2[np.ix_(I, I)] += (self.d2N * wq) @ self.d2N.T * h
            K1[np.ix_(I, I)] += (self.dN * wq) @ self.dN.T * h
            M[np.ix_(I, I)] += (self.N * wq) @ self.N.T * h
        self.K2, self.K1, self.M, self.nd = K2, K1, M, nd
        last = 2 * ne
        self.free_w = np.setdiff1d(np.arange(nd), [0, 1, last, last + 1])   # w = w' = 0
        self.free_u = np.setdiff1d(np.arange(nd), [0, last])                # u = 0

    def _load(self, f, fu, U, mu):
        b = np.zeros(self.nd)
        J = np.zeros((self.nd, self.nd))
        h = self.h
        for e in range(self.ne):
            I = np.arange(2 * e, 2 * e + 4)
            xq = self.xn[e] + self.tq * h
            uq = U[I] @ self.N
            b[I] += (self.N * self.wq) @ (mu * f(xq, uq)) * h
            J[np.ix_(I, I)] += (self.N * (self.wq * mu * fu(xq, uq))) @ self.N.T * h
        return b, J

    def solve(self, f: Fun, fu: Fun, mu: float = 1.0, U0=None, tol=1e-12, maxit=60):
        a2 = self.alpha**2
        fw, fu_ = self.free_w, self.free_u
        nw = len(fw)
        W = np.zeros(self.nd)
        U = np.zeros(self.nd) if U0 is None else np.array(U0, dtype=float)
        U[[0, 2 * self.ne]] = 0.0
        H = self.K1 + a2 * self.M
        prev = np.inf
        for it in range(maxit):
            b, Jf = self._load(f, fu, U, mu)
            R1 = (self.K2 @ W + b)[fw]
            R2 = (H @ U + self.M @ W)[fu_]
            J = np.block([[self.K2[np.ix_(fw, fw)], Jf[np.ix_(fw, fu_)]],
                          [self.M[np.ix_(fu_, fw)], H[np.ix_(fu_, fu_)]]])
            d = np.linalg.solve(J, -np.concatenate([R1, R2]))
            W[fw] += d[:nw]
            U[fu_] += d[nw:]
            step = max(np.max(np.abs(d[nw:])) / max(1.0, np.max(np.abs(U))),
                       np.max(np.abs(d[:nw])) / max(1.0, np.max(np.abs(W))))
            if step <= tol or (it >= 6 and step < 1e-8 and step > 0.5 * prev):
                break
            prev = step
        else:
            raise RuntimeError("finite-element Newton iteration did not converge")
        self.U, self.W = U, W
        return self.xn, U[0::2].copy()

    def evaluate(self, x, U=None):
        U = self.U if U is None else U
        x = np.atleast_1d(np.asarray(x, dtype=float))
        e = np.minimum((x / self.h).astype(int), self.ne - 1)
        t = (x - self.xn[e]) / self.h
        N, _, _ = hermite(t, self.h)
        return sum(U[2 * e + k] * N[k] for k in range(4))

    def nodal_initial_guess(self, ufun, dufun):
        """Hermite degrees of freedom (u, u') interpolating a given function."""
        U = np.zeros(self.nd)
        U[0::2] = ufun(self.xn)
        U[1::2] = dufun(self.xn)
        return U
