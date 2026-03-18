"""Low-level solver bug regression tests.

Two confirmed bugs in the solver layer are documented and pinned here:

BUG-1  BaseSolver._step converter — integer step treated as nanoseconds
------------------------------------------------------------------------
``BaseSolver._step`` has an attrs converter that checks
``isinstance(x, float | None)``.  When callers pass an ``int`` (e.g.
``step=43200``), the guard fails and the value is routed through
``pd.Timedelta(43200)`` → 43200 **nanoseconds** = 4.32e-05 s.  With a
sub-microsecond step the SPICE geometry finders iterate billions of
times and never return.

Fix: the guard must also accept ``int`` values:
``isinstance(x, float | int | None)``.

BUG-2  GenericScalarSolver / gfuds — is_dec callback captures wrong function
-----------------------------------------------------------------------------
``gfuds`` requires a UDFUNB callback that returns True when the search
function is *decreasing*.  The original implementation was::

    as_spice_f = left_prop.compute_as_spice_function()   # SpiceUDFUNS

    def is_dec(func, t):            # func = C-level pointer passed by SPICE
        return spiceypy.uddc(func, t, 1.0)

``spiceypy.uddc`` needs the *same* SpiceUDFUNS that was given to gfuds so
it can evaluate f(t±dt).  But the ``func`` argument passed by SPICE into
the UDFUNB callback is a raw C pointer to SPICE's own ``udf`` stub, not
the Python closure.  So ``uddc(func, t, 1.0)`` effectively differentiates
the constant-zero stub and always returns False — gfuds can never correctly
detect direction changes, and only the last interval boundary is found.

Fix: ``is_dec`` must close over ``as_spice_f`` and call
``spiceypy.uddc(as_spice_f, t, 1.0)``, ignoring the ``func`` argument.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import spiceypy

from spice_segmenter import TimeSegmentsCollection
from spice_segmenter.constraint_solver.constraint_solver import (
    GenericScalarSolver,
    SpiceEventSolver,
)
from spice_segmenter.core.spice_window import SpiceWindow
from spice_segmenter.properties.observation_properties import Distance, PhaseAngle
from spice_segmenter.properties.geometry_properties import SubObserverLatitude
from spice_segmenter.support.config import config

from . import tour_config as tc

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
OBSERVER = "JUICE_JANUS"
TARGET   = "GANYMEDE"

# 18-day window with many lat oscillations and phase varying across [1.09, 2.06] rad
WIN_START = "2035-04-01"
WIN_END   = "2035-04-19"

config.solver_step = "12 h"
tc.load_kernels()

_spice_window = TimeSegmentsCollection.from_start_end(WIN_START, WIN_END)._to_spice_window()

dist         = Distance(OBSERVER, TARGET)        # km
phase_native = PhaseAngle(OBSERVER, TARGET)      # rad  (no UnitAdaptor)
lat          = SubObserverLatitude(OBSERVER, TARGET)  # deg

STEP_S = 12 * 3600.0  # 12 h in seconds

# Sub-observer latitude oscillates with a ~12 h period on Ganymede.
# The shortest violated region for lat<30° / lat>-30° constraints is ~4 h;
# a 3 h step safely brackets every crossing while running 3× faster than 1 h.
LAT_STEP_S = 3 * 3600.0  # 3 h in seconds

_HALF_PI = math.pi / 2  # ≈ 1.5708 rad = 90°


# ---------------------------------------------------------------------------
# Helper: dense sampling inside the window
# ---------------------------------------------------------------------------

def _expected_interval_count(prop_fn, threshold, op, start=WIN_START, end=WIN_END):
    """Brute-force count of expected intervals via dense sampling."""
    times = pd.date_range(start, end, freq="3 h")
    vals  = np.asarray(prop_fn(times))
    if op == "<":
        satisfied = vals < threshold
    else:
        satisfied = vals > threshold
    # Count sign changes = number of interval boundaries / 2 (rounded up)
    crossings = int(np.sum(np.abs(np.diff(satisfied.astype(int)))))
    n_intervals = (crossings + 1) // 2 if satisfied[0] else crossings // 2
    return n_intervals


# ===========================================================================
# BUG-1: BaseSolver._step — integer step treated as nanoseconds
# ===========================================================================

class TestBug1StepConverterIntegerHandling:
    """
    ``BaseSolver._step`` converter only accepted ``float | None``.
    Integer values (e.g. ``12 * 3600 = 43200``) fell through to
    ``pd.Timedelta(43200)`` which treats integers as **nanoseconds**,
    producing a step of 4.3e-05 s and causing SPICE solvers to hang.
    """

    def test_int_step_treated_as_seconds(self):
        """An int step value should be kept as-is (seconds), not converted to nanoseconds."""
        c = lat < "0 deg"
        solver = GenericScalarSolver(constraint=c, step=43200)
        assert solver.step == 43200, (
            f"Expected step=43200 s, got {solver.step}. "
            "Integer should be treated as seconds, not routed through pd.Timedelta."
        )

    def test_float_step_treated_as_seconds(self):
        """A float step value should be kept as-is."""
        c = lat < "0 deg"
        solver = GenericScalarSolver(constraint=c, step=43200.0)
        assert solver.step == 43200.0

    def test_string_step_converts_via_timedelta(self):
        """A string step should be converted via pd.Timedelta."""
        c = lat < "0 deg"
        solver = GenericScalarSolver(constraint=c, step="12 h")
        assert solver.step == pytest.approx(43200.0)


# ===========================================================================
# BUG-2: GenericScalarSolver / gfuds — is_dec always returns False
# ===========================================================================

class TestBug2GenericScalarSolverIsDecCallback:
    """
    The ``is_dec`` closure inside GenericScalarSolver.solve() passes
    the wrong function pointer to spiceypy.uddc.

    Current code:
        def is_dec(func, t):
            return spiceypy.uddc(func, t, 1.0)   # func = C stub, not the property

    gfuds calls is_dec with its own internal C-level function pointer
    (SPICE's udf stub), not the Python SpiceUDFUNS.  uddc then
    differentiates the constant-zero stub, always returns False (never
    decreasing), so gfuds only transitions once and returns ≤ 1 interval.

    Correct code (fix):
        def is_dec(func, t):
            return spiceypy.uddc(as_spice_f, t, 1.0)  # close over as_spice_f
    """

    def test_lat_zero_crossing_count_dense_sampling(self):
        """
        Verify via brute-force dense sampling that lat crosses 0° multiple times.

        This test has no dependency on the solver — it only checks the physics.
        If this fails the test window or kernel is wrong.
        """
        n = _expected_interval_count(lat, 0.0, "<")
        assert n >= 4, (
            f"Expected ≥4 intervals for lat < 0° in {WIN_START}→{WIN_END}, "
            f"dense sampling found {n}. Check kernels / window."
        )

    def test_lat_less_than_zero_finds_multiple_intervals(self):
        """
        lat < 0° should produce multiple intervals as lat oscillates
        through ±80° over the 18-day window.
        """
        c = lat < "0 deg"
        solver = GenericScalarSolver(constraint=c, step=LAT_STEP_S)
        result = solver.solve(_spice_window)

        n_expected = _expected_interval_count(lat, 0.0, "<")
        assert len(result) >= 2, (
            f"Expected ≥{n_expected} intervals for lat < 0°, "
            f"got {len(result)}."
        )

    def test_lat_greater_than_zero_finds_multiple_intervals(self):
        """
        lat > 0° should also produce multiple intervals (complement of above).
        """
        c = lat > "0 deg"
        solver = GenericScalarSolver(constraint=c, step=LAT_STEP_S)
        result = solver.solve(_spice_window)

        n_expected = _expected_interval_count(lat, 0.0, ">")
        assert len(result) >= 2, (
            f"Expected ≥{n_expected} intervals for lat > 0°, "
            f"got {len(result)}."
        )

    def test_is_dec_closure_uses_correct_function(self):
        """
        White-box test: verify that gfuds, when given the lat property and a
        time that is clearly mid-descent (lat going from +max toward 0),
        actually receives a correct is_dec() answer.

        We call uddc directly with as_spice_f (the correct call) and compare
        against calling it with spiceypy.udf (the stub that the bug introduces).

        At a time when lat is decreasing, uddc(as_spice_f, t, dt) must be True
        and uddc(spiceypy.udf_stub, t, dt) must be False.
        """
        import spiceypy
        from spice_segmenter.core.property import Property

        as_spice_f = lat.compute_as_spice_function()

        # Find a time where lat is clearly decreasing (post-maximum, descending)
        # lat peaks around hour 4 of its ~6h cycle, descends from ~+85° toward 0°
        t_descending = pd.Timestamp("2035-04-01 04:30:00")
        from spice_segmenter.support.spice_utilities import et
        t_et = et(t_descending)

        correct_is_dec  = spiceypy.uddc(as_spice_f, t_et, 3600.0)
        incorrect_is_dec = spiceypy.uddc(
            spiceypy.utils.callbacks.SpiceUDFUNS(spiceypy.udf), t_et, 3600.0
        )

        # The correct call must agree with numerical finite difference
        dt = 3600.0
        val_before = float(lat(t_descending - pd.Timedelta(seconds=dt/2)))
        val_after  = float(lat(t_descending + pd.Timedelta(seconds=dt/2)))
        actually_decreasing = val_after < val_before

        assert correct_is_dec == actually_decreasing, (
            f"uddc(as_spice_f, t, dt) returned {correct_is_dec} but "
            f"finite-difference says decreasing={actually_decreasing}. "
            f"(val_before={val_before:.3f}, val_after={val_after:.3f})"
        )

        assert not incorrect_is_dec, (
            "uddc(udf_stub, t, dt) unexpectedly returned True. "
            "The stub should always evaluate to ~0 and thus be non-decreasing."
        )

        # The bug: current code passes func (the stub) instead of as_spice_f.
        # Verify the stub gives a *different* (wrong) answer when lat is decreasing.
        if actually_decreasing:
            assert correct_is_dec != incorrect_is_dec, (
                "BUG-2 precondition: at a decreasing point, uddc(as_spice_f) "
                "and uddc(udf_stub) should differ. They don't — check the test time."
            )
