"""Regression tests for property numerical outputs.

For each (observer, target, time) combination ``compute_all`` is called and
every computed scalar / vector value is compared against the stored baseline.

Adding a new case
-----------------
1. Add a new entry to ``CASES`` below.
2. Run ``pytest tests/regression/test_property_regression.py --update-regression``
   to generate the baseline.
3. Commit the new ``.yaml`` file alongside the test change.

Updating after a kernel delivery
---------------------------------
::

    pytest tests/regression/test_property_regression.py --update-regression

Review the git diff on the ``.yaml`` files to confirm only expected values
changed, then commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import tour_config as tc

tc.load_kernels()

from spice_segmenter.collections.snapshot import compute_all

# ---------------------------------------------------------------------------
# Test cases: (label, observer, target, time_iso, occultors)
# ---------------------------------------------------------------------------
# Times are chosen to give geometrically interesting configurations:
#   t_start  — near start of tour coverage
#   t_mid    — roughly mid-tour
#   t_end    — near end of tour coverage

_START, _END = tc.coverage
_T_START = str(_START + np.timedelta64(1, "D"))     # 1 day after coverage start
_T_MID   = str(_START + (_END - _START) // 2)       # midpoint
_T_END   = str(_END   - np.timedelta64(1, "D"))      # 1 day before coverage end

CASES: list[tuple[str, str, str, str, list[str] | None]] = [
    # label                             observer         target      time       occultors
    ("juice_janus_ganymede_start",     "JUICE_JANUS",  "GANYMEDE", _T_START,  ["JUPITER", "IO", "EUROPA", "CALLISTO"]),
    ("juice_janus_ganymede_mid",       "JUICE_JANUS",  "GANYMEDE", _T_MID,    ["JUPITER", "IO", "EUROPA", "CALLISTO"]),
    ("juice_janus_ganymede_end",       "JUICE_JANUS",  "GANYMEDE", _T_END,    ["JUPITER", "IO", "EUROPA", "CALLISTO"]),
    ("juice_janus_callisto_mid",       "JUICE_JANUS",  "CALLISTO", _T_MID,    ["JUPITER", "GANYMEDE", "IO", "EUROPA"]),
    ("juice_janus_europa_mid",         "JUICE_JANUS",  "EUROPA",   _T_MID,    ["JUPITER", "GANYMEDE", "IO", "CALLISTO"]),
    ("juice_janus_jupiter_mid",        "JUICE_JANUS",  "JUPITER",  _T_MID,    None),
]


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,observer,target,time,occultors", CASES, ids=[c[0] for c in CASES])
def test_properties(label, observer, target, time, occultors, regression_baseline):
    """Compute all auto-instantiable properties and compare against baseline."""
    snap = compute_all(
        observer, target, time,
        occultors=occultors,
    )

    # Flatten: include both computed values and errors (errors stored as
    # their string message so we can detect if a property that previously
    # errored now succeeds or vice-versa).
    actual: dict = {}
    for k, v in snap.values.items():
        actual[k] = v
    for k, msg in snap.errors.items():
        actual[f"__error__{k}"] = msg

    regression_baseline(
        f"properties/{label}.yaml",
        actual,
        # Relative tolerance 1 ppm for all numeric values.
        # Phase angles and angular quantities are inherently stable; distances
        # may drift by ~1 ppm between kernel versions.
        tolerances={"*": 1e-6},
    )
