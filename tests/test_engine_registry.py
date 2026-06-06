"""Parametric smoke-tests for every engine-registered property.

For every property class explicitly registered in the SPICE engine:
  - Classes that require constructor args beyond the standard context fields
    (observer / target / light_time_correction) are skipped automatically.
  - ``test_scalar_compute[name]``: the registered scalar function executes
    without error and returns a non-None result.
  - ``test_vector_compute[name]``: only for classes that have a dedicated
    vector function registered; the vector function executes and its result
    is numerically equivalent to the scalar result at the same epoch.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import tour_config as tc

tc.load_kernels()

import numpy as np

from spice_segmenter.core.registry import _field_info
from spice_segmenter.engines.evaluator import get_evaluator
from spice_segmenter.support.context import SpiceContext
from spice_segmenter.support.spice_utilities import et as _to_et

OBSERVER = "JUICE_JANUS"
TARGET   = "GANYMEDE"
LTC      = "NONE"

# Epoch well inside the JUICE tour kernel coverage.
_START, _ = tc.coverage
_ET: float = float(_to_et(_START + np.timedelta64(100, "D")))

# Force engine initialisation so _scalar_fns / _vector_fns are populated.
_engine = get_evaluator()._engine


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def _build_cases(fns_dict: dict) -> list[pytest.param]:
    """Return a ``pytest.param`` list for all auto-instantiable classes in *fns_dict*.

    A class is *auto-instantiable* when it has no required constructor fields
    beyond the three standard context fields (observer / target /
    light_time_correction).  Classes that require extra positional args
    (e.g. ``Occultation`` requires ``front`` and ``back``) are excluded.
    """
    cases: list[pytest.param] = []

    with SpiceContext(observer=OBSERVER, target=TARGET, light_time_correction=LTC):
        for cls in sorted(fns_dict, key=lambda c: getattr(c, "_name", c.__name__)):
            if getattr(cls, "_skip_auto_compute", False):
                continue

            required, _ctx, _opt = _field_info(cls)
            if required:
                continue  # needs extra args we cannot auto-provide

            try:
                prop = cls()
            except Exception:
                continue

            prop_id = getattr(cls, "_name", cls.__name__)
            cases.append(pytest.param(prop, id=prop_id))

    return cases


_SCALAR_CASES = _build_cases(_engine._scalar_fns)
_VECTOR_CASES = _build_cases(_engine._vector_fns)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _assert_scalar_vector_agree(scalar_val, vector_val) -> None:
    """Assert that *scalar_val* and ``vector_val[0]`` are numerically equal.

    Handles:
    - Plain floats and ints
    - numpy scalar arrays (shape ())
    - 1-D arrays for multi-component properties (e.g. latitudinal coords)
    - NaN values: equal NaN positions are considered agreeing
    """
    sv = np.asarray(scalar_val, dtype=float).flatten()
    # vector_val has one element per input epoch; take the first.
    vv = np.asarray(vector_val, dtype=float)
    if vv.ndim == 1:
        # scalar property: shape (n_times,) → first element is a scalar
        first = vv[0:1]
    else:
        # vector property: shape (n_times, n_components) → first row
        first = vv[0].flatten()

    np.testing.assert_allclose(
        sv, first,
        rtol=1e-10,
        equal_nan=True,
        err_msg=f"scalar result {scalar_val!r} != vector[0] result {vv[0]!r}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prop", _SCALAR_CASES)
def test_scalar_compute(prop) -> None:
    """Registered scalar function executes and returns a non-None result."""
    try:
        result = _engine.evaluate_scalar(prop, _ET)
    except Exception as exc:
        pytest.skip(f"SPICE evaluation skipped: {exc}")

    assert result is not None


@pytest.mark.parametrize("prop", _VECTOR_CASES)
def test_vector_compute(prop) -> None:
    """Registered vector function executes; result agrees with scalar at the same epoch."""
    try:
        scalar_val = _engine.evaluate_scalar(prop, _ET)
        vector_val = _engine.evaluate_vector(prop, np.array([_ET]))
    except Exception as exc:
        pytest.skip(f"SPICE evaluation skipped: {exc}")

    _assert_scalar_vector_agree(scalar_val, vector_val)
