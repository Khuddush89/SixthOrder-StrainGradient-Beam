# Verification

Run `python Green_Verification.py` for exact kernel identities followed by the numerical test suite. A failed check propagates a nonzero process exit status. Run `python code/scripts/audit_revision.py` for the independent audit.

| Check | Evidence |
|---|---|
| Green formula | 14 exact residual identities, followed by 60-digit rational-point evaluations |
| Fourth-order kernel | Exact third-derivative jump of 1 with normalization `1/(6*L^3)` |
| Numerical kernels | Agreement with split positive convolution at small and large parameters |
| Earlier numerical failures | Regression tests at `alpha=0.001` and `alpha=800` |
| Constants | Independent 65-digit evaluations of `A`, `B` and `eta_H` |
| Uniform loading | Exact equation, all endpoint conditions and asymptotic coefficients |
| Manufactured solution | Asymmetric positive nonconstant load at `L=1.7`, `alpha=2` |
| Convergence | Observed finite-difference and mixed finite-element orders |
| Folds | 48-to-64-point comparison and discrete equation residuals |
| Media | Animated GIF frames, original generation scripts, local links and opaque white PNG exports |

The local numerical suite contains 17 tests. GitHub Actions repeats the numerical and symbolic checks on Python 3.11 and 3.12 after the package is uploaded.

The former Maple worksheet is omitted because its stored output contained a failed jump check. The exact identities are provided in a runnable, independently verified symbolic script. Random sampling is documented as numerical evidence; it is not presented as a positivity proof.

The optimized cone constant is not used as a rigorous lower bound. Numerical folds are not interpreted as global nonexistence limits. These distinctions are maintained in the notes, results tables and animated captions.

See `verification/` for the actual local logs and `code/results/audit.json` for the independent audit values. Regeneration logs are machine-dependent; the scientific quantities are the reproducibility targets.
