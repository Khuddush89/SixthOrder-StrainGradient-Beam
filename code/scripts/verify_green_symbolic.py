#!/usr/bin/env python3
"""
Symbolic verification (SymPy) of Theorem 2.1: the twelve coefficients of the
Green's function G0, exactly as printed in the paper, satisfy

  * L0 G0 = G0^(6) - alpha^2 G0^(4) = 0 for x != s,
  * the six boundary conditions (2.1),
  * continuity of G0, ..., G0^(4) at x = s and the jump G0^(5)(s+) - G0^(5)(s-) = 1.

Each residual is first simplified to an exact symbolic zero and then evaluated
at random rational points with 60-digit arithmetic.  The script also confirms the normalisation 1/(6L^3) of the clamped
kernel G4 (the jump of its third derivative must be 1).

    python scripts/verify_green_symbolic.py
"""
import random
import sys

import sympy as sp

x, s, L, a = sp.symbols('x s L alpha', positive=True)
A1 = -(L**2 - 2*L*s + s**2)*s/(a**4*L**2)
B1 = (L**3 - 3*L*s**2 + 2*s**3)/(a**4*L**3)
C1 = -(L**2 - 2*L*s + s**2)*s/(2*a**2*L**2)
D1 = (L**3 - 3*L*s**2 + 2*s**3)/(6*a**2*L**3)
E1 = -(a*s*(L - s)*((L - s)*sp.exp(-a*L) - s) + L**2*sp.sinh(a*(L - s)))/(2*L**2*a**5*sp.sinh(a*L))
F1 = (a*s*(L - s)*((L - s)*sp.exp(a*L) - s) + L**2*sp.sinh(a*(L - s)))/(2*L**2*a**5*sp.sinh(a*L))
A2 = (a**2*L**2*s + 12*L - 6*s)*s**2/(6*a**4*L**2)
B2 = -(a**2*L**3 + 6*L - 4*s)*s**2/(2*a**4*L**3)
C2 = (2*L - s)*s**2/(2*a**2*L**2)
D2 = -(3*L - 2*s)*s**2/(6*a**2*L**3)
E2 = -(a*s*(L - s)*((L - s)*sp.exp(-a*L) - s) - L**2*sp.exp(-a*L)*sp.sinh(a*s))/(2*L**2*a**5*sp.sinh(a*L))
F2 = (a*s*(L - s)*((L - s)*sp.exp(a*L) - s) - L**2*sp.exp(a*L)*sp.sinh(a*s))/(2*L**2*a**5*sp.sinh(a*L))
GL = A1 + B1*x + C1*x**2 + D1*x**3 + E1*sp.exp(a*x) + F1*sp.exp(-a*x)
GR = A2 + B2*x + C2*x**2 + D2*x**3 + E2*sp.exp(a*x) + F2*sp.exp(-a*x)


def d(f, k):
    return sp.diff(f, x, k)


checks = {
    'equation, x < s': d(GL, 6) - a**2*d(GL, 4),
    'equation, x > s': d(GR, 6) - a**2*d(GR, 4),
    'G(0) = 0': GL.subs(x, 0),
    "G''(0) = 0": d(GL, 2).subs(x, 0),
    "G'''(0) - a^2 G'(0) = 0": (d(GL, 3) - a**2*d(GL, 1)).subs(x, 0),
    'G(L) = 0': GR.subs(x, L),
    "G''(L) = 0": d(GR, 2).subs(x, L),
    "G'''(L) - a^2 G'(L) = 0": (d(GR, 3) - a**2*d(GR, 1)).subs(x, L),
}
for k in range(5):
    checks[f'continuity of d^{k}G at x = s'] = (d(GR, k) - d(GL, k)).subs(x, s)
checks['jump of d^5 G at x = s equals 1'] = (d(GR, 5) - d(GL, 5)).subs(x, s) - 1

random.seed(1)
pts = [(sp.Rational(random.randint(1, 99), 100), sp.Rational(random.randint(1, 30), 10),
        sp.Rational(random.randint(5, 30), 10)) for _ in range(6)]
ok = True
for name, expr in checks.items():
    exact = sp.simplify(sp.expand(expr.rewrite(sp.exp))) == 0
    worst = 0
    for (ss, aa, LL) in pts:
        ss = ss * LL
        for xx in (ss / 3, (ss + LL) / 2):
            worst = max(worst, abs(sp.N(expr.subs({s: ss, a: aa, L: LL, x: xx}), 60)))
    flag = exact and worst < 1e-40
    ok &= flag
    print(f'{name:36s} exact identity: {exact}; max |residual| = {float(worst):.1e}  {"ok" if flag else "FAIL"}')

xi = sp.symbols('xi', positive=True)
K1 = xi**2*(L - s)**2*(3*s*L - 2*s*xi - L*xi)/(6*L**3)
K2 = s**2*(L - xi)**2*(3*xi*L - 2*s*xi - L*s)/(6*L**3)
jump = sp.simplify((sp.diff(K2, xi, 3) - sp.diff(K1, xi, 3)).subs(xi, s))
print(f'{"G4 with 1/(6L^3): jump of d^3 G4":36s} = {jump}  {"ok" if jump == 1 else "FAIL"}')
ok &= jump == 1
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
