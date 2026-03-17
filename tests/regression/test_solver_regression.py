"""Regression tests for constraint-solving (event-finding) results.

Each test defines a constraint, a search window, and a step size, then:

1. Solves the constraint to obtain a ``SpiceWindow``.
2. Serialises the result as a dict with:

   * ``n_intervals``      – number of solution intervals
   * ``total_duration_s`` – sum of interval lengths in seconds
   * ``intervals``        – list of ``{start, end}`` ISO-8601 strings
     (only the *first* ``MAX_INTERVALS_STORED`` are stored to keep files
     small, but ``n_intervals`` and ``total_duration_s`` cover the full
     window)

3. Compares against the stored baseline with configurable tolerances.

Tolerances
----------
* ``n_intervals``      – must match exactly (integer).
* ``total_duration_s`` – relative tolerance 1e-4 (~8 s per day of coverage).
* ``start`` / ``end`` of each interval – 60-second absolute tolerance (kernel
  updates may shift event boundaries by a few seconds).

Adding new solver cases
-----------------------
Append to ``SOLVER_CASES`` and run ``--update-regression``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests import tour_config as tc

tc.load_kernels()

from spice_segmenter import (
    Distance,
    Occultation,
    OccultationTypes,
    PhaseAngle,
    SpiceContext,
    TimeSegmentsCollection,
    config,
)
from spice_segmenter.properties.geometry_properties import (
    SubObserverLatitude,
    SubObserverIncidenceAngle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INTERVALS_STORED = 20   # cap stored intervals to keep YAML files compact

# Use a 1-year window during the Jupiter tour where close approaches occur.
# 2032 is representative: multiple Ganymede and Callisto flybys in this period.
SEARCH_WINDOW = TimeSegmentsCollection.from_start_end("2032-01-01T00:00:00", "2033-01-01T00:00:00")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _et_to_iso(et_val: float) -> str:
    """Convert SPICE ET to ISO-8601 string (second precision)."""
    return utc(et_val)


def _window_to_dict(window: TimeSegmentsCollection) -> dict[str, Any]:
    """Serialise a TimeSegmentsCollection to a plain dict suitable for YAML storage."""
    df = window.to_pandas(round_to=None)

    n = len(window)
    total_s: float = 0.0
    for iw in window:
        total_s += (iw.end - iw.start).total_seconds()

    intervals: list[dict[str, str]] = []
    for _, row in df.head(MAX_INTERVALS_STORED).iterrows():
        intervals.append({
            "start": str(row["start"]),
            "end":   str(row["end"]),
        })

    return {
        "n_intervals":       n,
        "total_duration_s":  round(total_s, 3),
        "intervals":         intervals,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
# Solver case definitions
# ---------------------------------------------------------------------------

# Each entry: (label, constraint_factory, step_s)
# constraint_factory() is called inside the test (after kernels are loaded)
# so SPICE refs are resolved at call time.

def _make_cases():
    return [
        (
            "distance_ganymede_lt_1Mkm",
            lambda: Distance("JUICE_JANUS", "GANYMEDE") < "1000000 km",
            6 * 3600,       # 6-hour step
        ),
        (
            "distance_callisto_lt_500k_km",
            lambda: Distance("JUICE_JANUS", "CALLISTO") < "500000 km",
            6 * 3600,
        ),
        (
            "phase_angle_ganymede_lt_90deg",
            lambda: PhaseAngle("JUICE_JANUS", "GANYMEDE") < "90 deg",
            12 * 3600,
        ),
        (
            "sub_sc_latitude_ganymede_gt_10deg",
            lambda: SubObserverLatitude("JUICE_JANUS", "GANYMEDE") > "10 deg",
            6 * 3600,
        ),
        (
            "sub_observer_incidence_ganymede_lt_90deg",
            lambda: SubObserverIncidenceAngle("JUICE_JANUS", "GANYMEDE") < "90 deg",
            6 * 3600,
        ),
        (
            "occultation_callisto_by_jupiter_any",
            lambda: Occultation("JUICE_JANUS", "CALLISTO", "JUPITER") == OccultationTypes.ANY,
            3 * 3600,
        ),
        (
            "combined_distance_and_latitude",
            lambda: (
                (Distance("JUICE_JANUS", "GANYMEDE") < "2000000 km")
                & (SubObserverLatitude("JUICE_JANUS", "GANYMEDE") > "0 deg")
            ),
            6 * 3600,
        ),
    ]


SOLVER_CASES = _make_cases()


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,constraint_factory,step_s",
    SOLVER_CASES,
    ids=[c[0] for c in SOLVER_CASES],
)
def test_solver(label, constraint_factory, step_s, regression_baseline):
    """Solve a constraint and compare interval results against baseline."""
    saved_step = config.solver_step
    config.solver_step = step_s
    try:
        constraint = constraint_factory()
        window = constraint.solve(SEARCH_WINDOW)
    finally:
        config.solver_step = saved_step

    actual = _window_to_dict(window)

    path = f"solvers/{label}.yaml"

    regression_baseline(
        path,
        _flatten(actual),
        # n_intervals is an integer — use a tiny rel tol so pytest.approx
        # treats it as exact.  Timestamps are handled by _compare_numeric
        # automatically (60-second tolerance).
        tolerances={"n_intervals": 1e-9, "total_duration_s": 1e-4},
    )


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict/list to dot-path keys for the baseline fixture."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{full_key}.{i}"))
                else:
                    out[f"{full_key}.{i}"] = item
        else:
            out[full_key] = v
    return out
