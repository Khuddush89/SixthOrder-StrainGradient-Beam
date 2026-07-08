"""
Green's Function Verification Script
====================================
This script numerically verifies the explicit Green's function G0(x,s)
for the sixth-order operator L0 = D^6 - alpha^2 D^4 with boundary
conditions (0) = u(L)=0, u''(0)=u''(L)=0, u'''(0)-alpha²u'(0)=0,
u'''(L)-alpha²u'(L)=0.

It checks:
  - Nonnegativity of the clamped beam kernel G4.
  - Convolution identity G0 = ∫ G2 * G4.
  - Nonpositivity of G0 on random samples.
  - Dense grid search for maximum of G0.
  - Plots cross-section and 3D surface.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

# Set plotting style (fallback if seaborn not available)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

# ============================================================
# 1. Explicit Green's function G0 (from Theorem 3.1)
# ============================================================
def G0(x, s, alpha, L):
    """
    Explicit piecewise Green's function G0(x,s).
    """
    # Left coefficients (x <= s)
    A1 = -(L**2 - 2*L*s + s**2)*s/(alpha**4*L**2)
    B1 = (L**3 - 3*L*s**2 + 2*s**3)/(alpha**4*L**3)
    C1 = -(L**2 - 2*L*s + s**2)*s/(2*alpha**2*L**2)
    D1 = (L**3 - 3*L*s**2 + 2*s**3)/(6*alpha**2*L**3)

    E1 = -(
        alpha*s*(L-s)*((L-s)*np.exp(-alpha*L)-s)
        + L**2*np.sinh(alpha*(L-s))
    )/(2*L**2*alpha**5*np.sinh(alpha*L))

    F1 = (
        alpha*s*(L-s)*((L-s)*np.exp(alpha*L)-s)
        + L**2*np.sinh(alpha*(L-s))
    )/(2*L**2*alpha**5*np.sinh(alpha*L))

    # Right coefficients (x > s)
    A2 = ((alpha**2*L**2*s + 12*L - 6*s)*s**2)/(6*alpha**4*L**2)
    B2 = -((alpha**2*L**3 + 6*L - 4*s)*s**2)/(2*alpha**4*L**3)
    C2 = ((2*L - s)*s**2)/(2*alpha**2*L**2)
    D2 = -((3*L - 2*s)*s**2)/(6*alpha**2*L**3)

    E2 = -(
        alpha*s*(L-s)*((L-s)*np.exp(-alpha*L)-s)
        - L**2*np.exp(-alpha*L)*np.sinh(alpha*s)
    )/(2*L**2*alpha**5*np.sinh(alpha*L))

    F2 = (
        alpha*s*(L-s)*((L-s)*np.exp(alpha*L)-s)
        - L**2*np.exp(alpha*L)*np.sinh(alpha*s)
    )/(2*L**2*alpha**5*np.sinh(alpha*L))

    if x <= s:
        return (A1 + B1*x + C1*x**2 + D1*x**3 +
                E1*np.exp(alpha*x) + F1*np.exp(-alpha*x))
    else:
        return (A2 + B2*x + C2*x**2 + D2*x**3 +
                E2*np.exp(alpha*x) + F2*np.exp(-alpha*x))


# ============================================================
# 2. Clamped biharmonic kernel G4 (positive)
# ============================================================
def G4(xi, s, L):
    """
    Clamped beam Green's function for w'''' = delta_s.
    Known to be ≥ 0 on [0,L].
    """
    if xi <= s:
        return (xi**2 * (L-s)**2 * (3*L*s - 2*s*xi - L*xi)) / (6*L**3)
    else:
        return (s**2 * (L-xi)**2 * (3*L*xi - 2*s*xi - L*s)) / (6*L**3)


# ============================================================
# 3. Dirichlet kernel G2 (negative)
# ============================================================
def G2(x, xi, alpha, L):
    """
    Green's function for u'' - alpha^2 u = f, u(0)=u(L)=0.
    Strictly negative on (0,L).
    """
    mn = min(x, xi)
    mx = max(x, xi)
    return (-np.sinh(alpha*mn) * np.sinh(alpha*(L - mx)) /
            (alpha * np.sinh(alpha*L)))


# ============================================================
# 4. Main verification routines
# ============================================================
def main():
    np.random.seed(12345)

    print("\n" + "="*60)
    print("Green's Function Verification")
    print("="*60)

    # ---- 1. Check G4 >= 0 ----
    print("\nChecking G4 >= 0 on 1000 random samples ...")
    g4_ok = True
    for _ in range(1000):
        L = np.random.uniform(0.5, 3.0)
        s = np.random.uniform(0, L)
        xi = np.random.uniform(0, L)
        val = G4(xi, s, L)
        if val < -1e-13:
            print(f"Negative G4 found: {val:.2e} at xi={xi:.3f}, s={s:.3f}, L={L:.3f}")
            g4_ok = False
            break
    if g4_ok:
        print("✅ G4 >= 0 on 1000 random samples.")

    # ---- 2. Convolution identity ----
    print("\nVerifying convolution identity ...")
    conv_ok = True
    for _ in range(30):
        alpha = np.random.uniform(0.2, 5.0)
        L = np.random.uniform(0.5, 3.0)
        s = np.random.uniform(0, L)
        x = np.random.uniform(0, L)

        integrand = lambda xi: G2(x, xi, alpha, L) * G4(xi, s, L)
        val_conv, _ = quad(integrand, 0, L, epsabs=1e-12, epsrel=1e-12, limit=500)
        val_G0 = G0(x, s, alpha, L)
        diff = abs(val_conv - val_G0)

        if diff > 1e-8:
            print(f"Convolution mismatch! diff = {diff:.2e} at x={x:.3f}, s={s:.3f}")
            conv_ok = False
            break
    if conv_ok:
        print("✅ Convolution identity verified (all differences < 1e-8).")

    # ---- 3. G0 <= 0 on random samples ----
    print("\nChecking G0 <= 0 on 1000 random samples ...")
    sign_ok = True
    for _ in range(1000):
        alpha = np.random.uniform(0.2, 5.0)
        L = np.random.uniform(0.5, 3.0)
        s = np.random.uniform(0, L)
        x = np.random.uniform(0, L)
        val = G0(x, s, alpha, L)
        if val > 1e-10:
            print(f"Positive G0 found: {val:.2e} at x={x:.3f}, s={s:.3f}, alpha={alpha:.2f}, L={L:.3f}")
            sign_ok = False
            break
    if sign_ok:
        print("✅ G0 <= 0 on 1000 random samples.")

    # ---- 4. Dense grid search for maximum ----
    print("\nDense grid search for extrema of G0 ...")
    mx = -1e100
    mn = 1e100
    argmax = None
    argmin = None

    for alpha in np.linspace(0.2, 5.0, 20):
        for L in np.linspace(0.5, 3.0, 20):
            xs = np.linspace(0, L, 80)
            ss = np.linspace(0, L, 80)
            for x in xs:
                for s in ss:
                    val = G0(x, s, alpha, L)
                    if val > mx:
                        mx = val
                        argmax = (x, s, alpha, L)
                    if val < mn:
                        mn = val
                        argmin = (x, s, alpha, L)

    print(f"Maximum of G0 = {mx:.2e} at (x,s,alpha,L) = {argmax}")
    print(f"Minimum of G0 = {mn:.2e} at (x,s,alpha,L) = {argmin}")

    if mx <= 1e-10:
        print("✅ Dense-grid test confirms G0 <= 0.")
    else:
        print("⚠️ Possible positive values detected – please inspect.")

    # ---- 5. Plots ----
    L_fixed = 1.0
    alpha_fixed = 1.0
    s_fixed = 0.5

    # Cross-section plot
    xvals = np.linspace(0, L_fixed, 500)
    g0vals = [G0(x, s_fixed, alpha_fixed, L_fixed) for x in xvals]

    plt.figure(figsize=(8, 5))
    plt.plot(xvals, g0vals, color='#2E8B57', linewidth=2.5, label=r'$G_0(x, 0.5)$')
    plt.fill_between(xvals, g0vals, alpha=0.2, color='#2E8B57')
    plt.xlabel(r'$x$', fontsize=13)
    plt.ylabel(r'$G_0(x, 0.5)$', fontsize=13)
    plt.title(r'$G_0(x, s=0.5) \leq 0$', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3D surface plot
    n = 120
    xvals = np.linspace(0, L_fixed, n)
    svals = np.linspace(0, L_fixed, n)
    X, S = np.meshgrid(xvals, svals)
    Z = np.zeros_like(X)

    for i in range(n):
        for j in range(n):
            Z[i, j] = G0(X[i, j], S[i, j], alpha_fixed, L_fixed)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, S, Z,
        cmap='plasma',
        edgecolor='none',
        alpha=0.92,
        antialiased=True
    )

    ax.view_init(elev=28, azim=135)
    ax.set_xlabel(r'$x$', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$s$', fontsize=12, labelpad=10)
    ax.set_zlabel(r'$G_0$', fontsize=12, labelpad=10)
    ax.set_title(r'$G_0(x,s) \leq 0$ on $[0,1]^2$', fontsize=14, fontweight='bold', pad=15)

    # Remove colorbar and make panes transparent for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()
    plt.show()

    print("\nAll numerical tests and plots completed successfully!")


if __name__ == "__main__":
    main()