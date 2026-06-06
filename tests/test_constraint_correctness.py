"""Correctness tests for composite AND / OR constraints.

Strategy
--------
For every composite constraint under test we:

1. Solve it over a fixed window to obtain a ``TimeSegmentsCollection``.
2. Sample the *full* window densely (every 6 hours).
3. Evaluate every leaf property at each sample time to reconstruct the expected
   boolean value directly from physics.
4. Assert that

   * every sample that falls *inside* a solved segment evaluates to ``True``
     (no false positives — segments must not span times that violate the constraint).
   * every sample that falls *outside* all solved segments evaluates to ``False``
     (no false negatives — the solver must not miss valid intervals).

All constraints are chosen to produce many intervals (many oscillation cycles)
so that both AND and OR logic are exercised thoroughly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spice_segmenter import TimeSegmentsCollection
from spice_segmenter.properties.geometry_properties import SubObserverLatitude
from spice_segmenter.properties.observation_properties import Distance, PhaseAngle
from spice_segmenter.support.config import config

from . import tour_config as tc

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
# Short window for phase/lat single tests: ~18 days of dense GANYMEDE
# flyby activity where all kernels are fully covered.  SubObserverLatitude
# oscillates through ±80° several times, giving many crossings.
# Phase spans 1.09–2.06 rad in this window, crossing π/2 several times.
SINGLE_WIN_START = "2035-04-01"
SINGLE_WIN_END   = "2035-04-19"

# Longer window for distance single tests: dist varies from ~3 000 to
# ~2 900 000 km, crossing 50 000 km 7 times (many intervals).
DIST_WIN_START = "2034-06-01"
DIST_WIN_END   = "2035-04-19"

# Larger window for compound AND/OR tests: covers many orbital periods.
COMPOUND_WIN_START = "2032-04-01"
COMPOUND_WIN_END   = "2035-04-19"

SAMPLE_STEP = "6 h"   # dense enough to catch narrow intervals
SOLVER_STEP = "12 h"  # coarse enough for reasonable test speed

# Sub-observer latitude (and phase angle) oscillates with a ~12 h period on
# Ganymede.  The shortest "violated" region for lat<30° or lat>-30° is ~4 h.
# A 3 h step is below that threshold and therefore catches every crossing,
# while being 3× faster than a 1 h step over multi-year compound windows.
LAT_SOLVER_STEP = 3 * 3600.0   # 3 h in seconds
LAT_SAMPLE_STEP = "3 h"        # matches solver step; no sub-step resolution needed

# Some compound constraints have narrow violated regions (< 3 h) that a 3 h
# step would straddle and miss, producing false positives.
# Specifically: phase < 120° has a ~1 h violated region per cycle;
#               lat ∈ [-60°, 60°] has a ~1 h violated region near trough/peak.
# gfevnt is a C-level call, so a 1 h step here costs virtually nothing extra.
TIGHT_SOLVER_STEP = 1 * 3600.0  # 1 h in seconds
TIGHT_SAMPLE_STEP = "1 h"       # finer sampling to detect narrow false positives

config.solver_step = SOLVER_STEP
tc.load_kernels()

SINGLE_WINDOW   = TimeSegmentsCollection.from_start_end(SINGLE_WIN_START, SINGLE_WIN_END)
DIST_WINDOW     = TimeSegmentsCollection.from_start_end(DIST_WIN_START, DIST_WIN_END)
COMPOUND_WINDOW = TimeSegmentsCollection.from_start_end(COMPOUND_WIN_START, COMPOUND_WIN_END)

# Keep WINDOW as an alias for backward compat with compound tests.
WINDOW_START = COMPOUND_WIN_START
WINDOW_END   = COMPOUND_WIN_END
WINDOW       = COMPOUND_WINDOW

# ---------------------------------------------------------------------------
# Shared property instances
# ---------------------------------------------------------------------------
OBSERVER = "JUICE_JANUS"
TARGET   = "GANYMEDE"

dist  = Distance(OBSERVER, TARGET)                   # native unit: km
# phase_native: raw PhaseAngle — native unit is RADIANS, no UnitAdaptor.
# Use this for single-operator tests to avoid any unit conversion in the solver.
phase_native = PhaseAngle(OBSERVER, TARGET)          # native unit: rad
# phase_deg: converted to degrees — kept for compound tests / plotting.
phase = PhaseAngle(OBSERVER, TARGET).as_unit("deg")  # deg (UnitAdaptor)
lat   = SubObserverLatitude(OBSERVER, TARGET)        # native unit: deg

import math

_HALF_PI = math.pi / 2   # ≈ 1.5708 rad  (= 90°)
_TWO_PI  = 2 * math.pi   # ≈ 6.2832 rad  (= 360°)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_times(start: str = WINDOW_START, end: str = WINDOW_END, step: str = SAMPLE_STEP) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=step)


def _inside_mask(
    times: pd.DatetimeIndex,
    result: TimeSegmentsCollection,
) -> np.ndarray:
    """Boolean mask: True for each time that falls inside *any* solved segment."""
    # Normalise to tz-naive UTC so comparisons with Timestamp objects always work.
    times_naive = times.tz_localize(None) if times.tz is not None else times
    mask = np.zeros(len(times_naive), dtype=bool)
    for seg in result:
        seg_start = pd.Timestamp(seg.start).tz_localize(None) if seg.start.tzinfo else pd.Timestamp(seg.start)
        seg_end   = pd.Timestamp(seg.end).tz_localize(None)   if seg.end.tzinfo   else pd.Timestamp(seg.end)
        mask |= (times_naive >= seg_start) & (times_naive <= seg_end)
    return mask


def _verify(
    constraint,
    expected_fn,
    *,
    label: str,
    window: TimeSegmentsCollection = COMPOUND_WINDOW,
    sample_start: str = COMPOUND_WIN_START,
    sample_end: str   = COMPOUND_WIN_END,
    solver_step: float | None = None,
    sample_step: str = SAMPLE_STEP,
) -> None:
    """Solve *constraint*, sample densely, compare with *expected_fn*.

    Parameters
    ----------
    constraint:
        The ``ConstraintBase`` to solve.
    expected_fn:
        ``(dist_vals, phase_native_vals, lat_vals) -> np.ndarray[bool]`` —
        reproduces the expected truth value from **native-unit** property
        arrays (dist in km, phase in rad, lat in deg).
    label:
        Human-readable label used in assertion messages.
    window:
        Solving window (defaults to COMPOUND_WINDOW).
    sample_start / sample_end:
        Time range to sample for verification (defaults to compound window).
    solver_step:
        Override the solver step size in seconds for this test.
    sample_step:
        Frequency string for dense sampling (defaults to SAMPLE_STEP).
    """
    solve_kwargs = {}
    if solver_step is not None:
        solve_kwargs["step"] = solver_step
    result = constraint.solve(window, **solve_kwargs)
    assert len(result) >= 2, (
        f"{label}: expected ≥2 solved intervals (constraint may be too loose "
        f"or too tight for this window), got {len(result)}"
    )

    times            = _sample_times(sample_start, sample_end, step=sample_step)
    inside           = _inside_mask(times, result)
    dist_vals        = np.asarray(dist(times))
    phase_native_vals = np.asarray(phase_native(times))   # radians
    lat_vals         = np.asarray(lat(times))
    expected         = np.asarray(expected_fn(dist_vals, phase_native_vals, lat_vals), dtype=bool)

    # Samples inside a solved segment must satisfy the constraint
    false_positives = inside & ~expected
    assert not false_positives.any(), (
        f"{label}: {false_positives.sum()} sample(s) INSIDE a solved segment "
        f"do NOT satisfy the constraint (solver included times that should be excluded).\n"
        f"First offending times: {times[false_positives].tolist()[:5]}"
    )

    # Samples outside all solved segments must NOT satisfy the constraint
    false_negatives = ~inside & expected
    assert not false_negatives.any(), (
        f"{label}: {false_negatives.sum()} sample(s) OUTSIDE all solved segments "
        f"DO satisfy the constraint (solver missed valid intervals).\n"
        f"First offending times: {times[false_negatives].tolist()[:5]}"
    )


def _verify_single(constraint, expected_fn, *, label: str, solver_step: float | None = None, sample_step: str = SAMPLE_STEP) -> None:
    """Like _verify but uses the short, fully-covered SINGLE_WINDOW (phase/lat)."""
    _verify(
        constraint,
        expected_fn,
        label=label,
        window=SINGLE_WINDOW,
        sample_start=SINGLE_WIN_START,
        sample_end=SINGLE_WIN_END,
        solver_step=solver_step,
        sample_step=sample_step,
    )


def _verify_dist(constraint, expected_fn, *, label: str) -> None:
    """Like _verify but uses the DIST_WINDOW (distance oscillates widely here)."""
    _verify(
        constraint,
        expected_fn,
        label=label,
        window=DIST_WINDOW,
        sample_start=DIST_WIN_START,
        sample_end=DIST_WIN_END,
    )


# ---------------------------------------------------------------------------
# Single-operator (leaf) constraints — sanity baseline
#
# All properties are evaluated in their NATIVE units throughout:
#   dist          → km
#   phase_native  → rad  (PhaseAngle with no UnitAdaptor wrapper)
#   lat           → deg
#
# Thresholds are chosen in the same native units so no conversion can
# silently corrupt the comparison inside the solver.
# ---------------------------------------------------------------------------

def test_single_distance_less() -> None:
    """dist < 50 000 km (native: km) — 7 crossings in the dist window."""
    c = dist < "50000 km"

    def expected(d, p_rad, l):
        return d < 50_000

    _verify_dist(c, expected, label="dist<50000km")


def test_single_distance_greater() -> None:
    """dist > 50 000 km (native: km)."""
    c = dist > "50000 km"

    def expected(d, p_rad, l):
        return d > 50_000

    _verify_dist(c, expected, label="dist>50000km")


def test_single_phase_less() -> None:
    """phase_native < pi/2 rad  (≈ 90°) — native radians, no conversion.

    Phase oscillates with a ~6 h period on Ganymede (JUICE orbits rapidly),
    so the solver step must be shorter than half the oscillation period.
    """
    c = phase_native < f"{_HALF_PI} rad"

    def expected(d, p_rad, l):
        return p_rad < _HALF_PI

    _verify_single(c, expected, label="phase<pi/2_rad", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_single_phase_greater() -> None:
    """phase_native > pi/2 rad  (≈ 90°) — native radians, no conversion."""
    c = phase_native > f"{_HALF_PI} rad"

    def expected(d, p_rad, l):
        return p_rad > _HALF_PI

    _verify_single(c, expected, label="phase>pi/2_rad", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_single_lat_less() -> None:
    """lat < 0° (native: deg).

    Latitude oscillates with a ~6 h period on Ganymede, so the solver
    step must be shorter than half the oscillation period.
    """
    c = lat < "0 deg"

    def expected(d, p_rad, l):
        return l < 0

    _verify_single(c, expected, label="lat<0deg", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_single_lat_greater() -> None:
    """lat > 0° (native: deg)."""
    c = lat > "0 deg"

    def expected(d, p_rad, l):
        return l > 0

    _verify_single(c, expected, label="lat>0deg", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


# ---------------------------------------------------------------------------
# AND constraints
# ---------------------------------------------------------------------------

def test_and_distance_latitude() -> None:
    """(dist < 28 000 km) & (lat > -30°) & (lat < 30°)

    Reproduces the original issue: the solver should NOT include times where
    |lat| > 30°.
    """
    c = (dist < "28000 km") & (lat > "-30 deg") & (lat < "30 deg")

    def expected(d, p, l):
        return (d < 28_000) & (l > -30) & (l < 30)

    _verify(c, expected, label="AND(dist<28000, lat∈[-30,30])", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_and_distance_phase() -> None:
    """(dist < 40 000 km) & (phase < 90°)"""
    c = (dist < "40000 km") & (phase < "90 deg")

    def expected(d, p, l):
        return (d < 40_000) & (p < _HALF_PI)

    _verify(c, expected, label="AND(dist<40000, phase<90)", solver_step=TIGHT_SOLVER_STEP, sample_step=TIGHT_SAMPLE_STEP)


def test_and_three_properties() -> None:
    """(dist < 50 000 km) & (phase < 120°) & (lat > -60°) & (lat < 60°)"""
    c = (
        (dist < "50000 km")
        & (phase < "120 deg")
        & (lat > "-60 deg")
        & (lat < "60 deg")
    )

    def expected(d, p, l):
        return (d < 50_000) & (p < math.radians(120)) & (l > -60) & (l < 60)

    _verify(c, expected, label="AND(dist<50000, phase<120, lat∈[-60,60])", solver_step=TIGHT_SOLVER_STEP, sample_step=TIGHT_SAMPLE_STEP)


# ---------------------------------------------------------------------------
# OR constraints
# ---------------------------------------------------------------------------

def test_or_latitude_bands() -> None:
    """(lat > 30°) | (lat < -30°)

    Two symmetric polar bands — lat oscillates through both repeatedly,
    producing many intervals.
    """
    c = (lat > "30 deg") | (lat < "-30 deg")

    def expected(d, p, l):
        return (l > 30) | (l < -30)

    _verify(c, expected, label="OR(lat>30, lat<-30)", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_or_distance_phase() -> None:
    """(dist < 20 000 km) | (phase > 120°)"""
    c = (dist < "20000 km") | (phase > "120 deg")

    def expected(d, p, l):
        return (d < 20_000) | (p > math.radians(120))

    _verify(c, expected, label="OR(dist<20000, phase>120)", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


# ---------------------------------------------------------------------------
# Mixed AND + OR
# ---------------------------------------------------------------------------

def test_mixed_and_or() -> None:
    """(dist < 50 000 km) & ((lat > 30°) | (lat < -30°))

    AND of a scalar constraint with an OR sub-tree — exercises SpiceWindowSolver
    nesting with the two solver types combined.
    """
    c = (dist < "50000 km") & ((lat > "30 deg") | (lat < "-30 deg"))

    def expected(d, p, l):
        return (d < 50_000) & ((l > 30) | (l < -30))

    _verify(c, expected, label="AND(dist<50000, OR(lat>30, lat<-30))", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


def test_mixed_or_of_and() -> None:
    """((dist < 28 000 km) & (lat > 30°)) | ((dist < 28 000 km) & (lat < -30°))

    OR of two AND sub-trees — logically equivalent to the mixed AND+OR above
    but built from the opposite direction.
    """
    c = ((dist < "28000 km") & (lat > "30 deg")) | (
        (dist < "28000 km") & (lat < "-30 deg")
    )

    def expected(d, p, l):
        return ((d < 28_000) & (l > 30)) | ((d < 28_000) & (l < -30))

    _verify(c, expected, label="OR(AND(dist<28000,lat>30), AND(dist<28000,lat<-30))", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)


# ---------------------------------------------------------------------------
# Inversion / NOT
# ---------------------------------------------------------------------------

def test_inverted_constraint() -> None:
    """~(dist < 28 000 km)  ≡  dist ≥ 28 000 km"""
    c = ~(dist < "28000 km")

    def expected(d, p, l):
        return ~(d < 28_000)

    _verify(c, expected, label="NOT(dist<28000)")


def test_inverted_and() -> None:
    """~((dist < 28 000 km) & (lat > -30°) & (lat < 30°))"""
    c = ~((dist < "28000 km") & (lat > "-30 deg") & (lat < "30 deg"))

    def expected(d, p, l):
        return ~((d < 28_000) & (l > -30) & (l < 30))

    _verify(c, expected, label="NOT(AND(dist<28000, lat∈[-30,30]))", solver_step=LAT_SOLVER_STEP, sample_step=LAT_SAMPLE_STEP)
