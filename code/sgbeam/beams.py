r"""
Mid-span deflection of beams under a uniform load q (clamped ends), used for the
comparison of Section 6.

* Euler--Bernoulli:      w_EB = q L^4 / (384 E I)
* Timoshenko (shear coefficient k_s):
                         w_T  = q L^4 / (384 E I) + q L^2 / (8 k_s G A)
  Timoshenko's theory is used with Cowper's coefficient for a rectangular
  section, k_s = 10 (1 + nu) / (12 + 11 nu). The conventional 5/6 choice
  is compared in the same linear shear formula; it is not a separate
  implementation of a full Reissner model.
  For a rectangular section, E I / (k_s G A) = (1 + nu) h^2 / (6 k_s), so

      w_T / w_EB = 1 + 8 (1 + nu) (h/L)^2 / k_s .

* strain-gradient beam of the paper (classical part clamped, Remark 1.1):
                         w_SG / w_EB = D(L/g),
  with the size-effect factor D of :func:`sgbeam.green.size_effect_factor`.

Author: M. Khuddush.  Licence: MIT.
"""
from __future__ import annotations

import numpy as np

from .green import size_effect_factor


def k_cowper(nu: float = 0.3) -> float:
    """Cowper's shear coefficient of a rectangular cross-section."""
    return 10 * (1 + nu) / (12 + 11 * nu)


K_CONVENTIONAL = 5.0 / 6.0
K_REISSNER = K_CONVENTIONAL  # legacy name retained for import compatibility


def shear_ratio(L_over_h, nu: float = 0.3, ks: float | None = None):
    """w/w_EB for a clamped Timoshenko-type beam (rectangular section)."""
    ks = k_cowper(nu) if ks is None else ks
    r = 1.0 / np.asarray(L_over_h, dtype=float)
    return 1.0 + 8.0 * (1.0 + nu) * r**2 / ks


def gradient_ratio(L_over_g):
    """w/w_EB for the strain-gradient beam of the paper (= D(L/g))."""
    return size_effect_factor(L_over_g)


def balance_length(nu: float = 0.3, ks: float = K_CONVENTIONAL) -> float:
    """g/h for which the leading corrections 16 (g/L)^2 and 8(1+nu)(h/L)^2/k_s are equal."""
    return float(np.sqrt((1 + nu) / (2 * ks)))
