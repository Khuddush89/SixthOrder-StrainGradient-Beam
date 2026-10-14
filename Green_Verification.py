"""Compatible entry point for the verified numerical Green kernel.

Run this file to execute the symbolic identities and the numerical test suite.
The operational G0 uses positive quadrature or an overflow-free expression.
"""
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from sgbeam import green as gr


def G0(x, s, alpha=1.0, L=1.0):
    """Nonpositive kernel of D^6-alpha^2 D^4; accepts scalars or arrays."""
    if not np.isfinite(alpha) or not np.isfinite(L) or alpha <= 0 or L <= 0:
        raise ValueError("alpha and L must be finite and positive")
    xx, ss = np.broadcast_arrays(np.asarray(x, float), np.asarray(s, float))
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(ss)):
        raise ValueError("x and s must be finite")
    if np.any((xx < 0) | (xx > L) | (ss < 0) | (ss > L)):
        raise ValueError("x and s must lie in [0,L]")
    value = -gr.Gt(xx, ss, alpha, L)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("Nonfinite Green-kernel value")
    return float(value) if value.ndim == 0 else value


G2 = gr.G2
G4 = gr.G4


def main():
    commands = [
        [sys.executable, "code/scripts/verify_green_symbolic.py"],
        [sys.executable, "-m", "pytest", "-q", "code/tests"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print("Symbolic verification and numerical tests passed.")


if __name__ == "__main__":
    main()
