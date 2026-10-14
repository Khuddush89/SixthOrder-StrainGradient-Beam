# Model and interpretation

The numerical routines solve

$$-u^{(6)}+\alpha^2u^{(4)}=f(x,u),\quad 0<x<L,$$

under `u=u''=0` and `u'''-alpha^2*u'=0` at both endpoints. Set `p=alpha^2*u-u''`. Then `p''''=f`, `p=p'=0`, and `-u''+alpha^2*u=p` with zero Dirichlet values. This factorization gives the positive response kernel used by the solvers.

`G0` denotes the nonpositive kernel of the opposite-sign operator. `Gt=-G0` is the positive response kernel. Both use the stable numerical path. The raw coefficient formula is separately named `sgbeam.green.G0_raw` and is retained for algebra checks. The compatible top-level `Green_Verification.G0` uses `-Gt` and also validates its input domain.

## Boundary interpretation

The imposed constraints are not natural simply supported conditions derived from the strain-gradient energy. The generalized physical moment is proportional to `w''-g^2*w''''`. Setting the curvature to zero alone does not set that moment to zero. Under fixed physical loading, the imposed constraints approach the clamped classical beam as the material length tends to zero.

The Green kernel is generally nonsymmetric. Symmetry in the two arguments is not assumed in any solver or positivity argument.

## Constants and existence examples

`A` is the maximum integral of the positive kernel over the full interval. `B` is the minimum integral over the central interval. The explicit lower cone constant is `sinh(alpha*delta)/sinh(alpha*L)`. Optimized `eta_star` values are numerical estimates, not certified global bounds.

At `L=alpha=1`, `delta=1/4`, the analytically verified exponential annulus uses `eta=0.2`, `r=0.01`, `R=75`. At `alpha=5`, the quadratic example uses `eta=0.02`, `r=0.72`, `R=5600`. The wider outer radii are theoretical bounds; they are not the computed maximum deflections.

For the exponential law, the sufficient bound `0<mu<1/(e*A)` yields at least two positive solutions. Numerical continuation and fold locations do not exclude additional solution branches or establish a global nonexistence threshold.

## Mechanical scaling

For a physical coordinate `y`, beam length `ell`, deflection `w` and reference deflection `w_c`, set `x=y/ell`, `u=w/w_c`, `kappa=ell/g`. The exponential-load parameter is `mu=kappa^2*q0*ell^4/(S*w_c)`.

The size factor compares the maximum uniform-load deflection with the clamped Euler–Bernoulli maximum. It does not assert pointwise ordering of every profile. The shear benchmarks use Cowper and conventional `5/6` coefficients in the same linear shear formula.

The nonlinear laws illustrate the mathematical theory. No experimental calibration or external dataset is claimed. Animations sweep static model parameters or camera angles; they do not solve a transient beam equation.
