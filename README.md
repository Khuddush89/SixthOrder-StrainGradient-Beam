# Sixth-Order Strain-Gradient Beam Deflection Problem

This repository contains the Maple and Python programs accompanying the research paper

> **Positive Solutions for a Sixth-Order Strain-Gradient Beam Deflection Problem with Generalized Simply Supported Boundary Conditions**

## Description

The repository provides symbolic and numerical computations used in the paper, including

- symbolic derivation of the Green's function;
- verification of all boundary and jump conditions;
- positivity verification of the Green's function;
- computation of the constants
  - \(A\),
  - \(B\),
  - \(\eta\);
- numerical verification of the hypotheses of Krasnosel'skii's fixed point theorem;
- reproduction of the numerical example presented in the paper.

## Repository Structure

```
Maple/
    GreenFunctionDerivation.mw
    VerifyGreenFunction.mw
    NumericalExample.mw

Python/
    positivity_check.py
    compute_constants.py
    numerical_example.py
```

## Software Requirements

- Maple 2023 or later
- Python 3.11+
- NumPy
- SciPy
- Matplotlib

## Reproducing the Results

1. Run the Maple worksheets to derive and verify the Green's function.
2. Execute the Python scripts to
   - verify positivity,
   - compute \(A\), \(B\), and \(\eta\),
   - reproduce the numerical example.

The numerical values reported in the manuscript are obtained for

\[
L=1,\qquad
\alpha=1,\qquad
\delta=\frac14.
\]

## Citation

If you use these codes, please cite

> M. Khuddush and co-authors,
> *Positive Solutions for a Sixth-Order Strain-Gradient Beam Deflection Problem with Generalized Simply Supported Boundary Conditions*.

(Complete bibliographic information will be added after publication.)

## License

This project is distributed under the MIT License.
