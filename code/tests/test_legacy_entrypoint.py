"""Regression checks for the previously unstable repository entry point."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sgbeam import green as gr

spec = importlib.util.spec_from_file_location("legacy", Path(__file__).parents[2] / "Green_Verification.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


@pytest.mark.parametrize("alpha,expected", [(0.001, -0.00045572912132626966), (800.0, -8.137717183430989e-9)])
def test_previously_failed_parameters(alpha, expected):
    actual = legacy.G0(0.5, 0.5, alpha, 1.0)
    assert np.isfinite(actual) and actual < 0
    assert np.isclose(actual, expected, rtol=1e-12)
    assert np.isclose(gr.G0(0.5, 0.5, alpha), expected, rtol=1e-12)


def test_invalid_parameters_are_rejected():
    for args in [(0.5, 0.5, 0, 1), (0.5, 0.5, np.nan, 1), (0.5, 0.5, 1, -1), (np.nan, 0.5, 1, 1)]:
        with pytest.raises(ValueError):
            legacy.G0(*args)
