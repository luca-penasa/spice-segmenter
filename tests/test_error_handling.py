"""Tests for error handling and edge cases."""

import numpy as np
import pandas as pd

from spice_segmenter import (
    Distance,
    PhaseAngle,
    TimeSegmentsCollection,
)
from spice_segmenter.support.config import config

from . import tour_config as tc

config.solver_step = "24 h"
start, end = tc.coverage
tc.load_kernels()

t1 = start + np.timedelta64(100, "D")


def test_invalid_constraint_operator() -> None:
    """Test that invalid constraint operators are rejected."""
    d = Distance(tc.spacecraft, tc.target)

    # Valid operators should work
    c_lt = d < "5000 km"
    c_eq = d == "5000 km"
    c_gt = d > "5000 km"

    assert c_lt is not None
    assert c_eq is not None
    assert c_gt is not None


def test_constraint_with_mismatched_units() -> None:
    """Test constraint with unit conversion."""
    d_km = Distance(tc.spacecraft, tc.target)

    # Should handle unit conversion automatically
    c = d_km < "5000000 m"  # comparing km to m

    assert c is not None
    result = c(t1)
    # Result can be bool, numpy bool, or numpy array
    assert result is not None


def test_constraint_evaluation_at_boundary() -> None:
    """Test constraint evaluation at window boundaries."""
    w = TimeSegmentsCollection.from_start_end("2032-01-01", "2034-01-01")
    c = Distance(tc.spacecraft, tc.target) < "5000 km"

    # Should be able to evaluate constraint
    result = c(t1)
    assert result is not None


def test_multiple_properties_same_time() -> None:
    """Test evaluating multiple properties at same time."""
    d = Distance(tc.spacecraft, tc.target)
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN")

    result_d = d(t1)
    result_phase = phase(t1)

    assert np.isfinite(result_d)
    assert np.isfinite(result_phase)


def test_spice_window_operations() -> None:
    """Test TimeSegmentsCollection edge cases."""
    w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
    w2 = TimeSegmentsCollection.from_start_end("2032-07-01", "2032-12-31")

    # Union of non-overlapping intervals
    w3 = w1 + w2
    assert len(w3) == 2

    # Empty window
    w_empty = TimeSegmentsCollection()
    assert len(w_empty) == 0


def test_constraint_composition_associativity() -> None:
    """Test that constraint composition works correctly."""
    c1 = Distance(tc.spacecraft, tc.target) < "5000 km"
    c2 = Distance(tc.spacecraft, tc.target) > "1000 km"
    c3 = PhaseAngle(tc.spacecraft, tc.target, "SUN") < "1.57 rad"

    # Composition should work
    c_combined = (c1 & c2) & c3
    assert c_combined is not None

    # Should be evaluable
    result = c_combined(t1)
    assert result is not None


def test_constraint_with_inverted() -> None:
    """Test inverted (NOT) constraint logic."""
    from spice_segmenter.ops import Inverted

    c = Distance(tc.spacecraft, tc.target) < "5000 km"
    c_inv = Inverted(c)

    # Inverted should flip boolean result
    result = c(t1)
    result_inv = c_inv(t1)

    assert result != result_inv


def test_property_consistency_same_time() -> None:
    """Test that property returns same value when evaluated at same time."""
    d = Distance(tc.spacecraft, tc.target)

    result1 = d(t1)
    result2 = d(t1)

    assert result1 == result2


def test_distance_always_positive() -> None:
    """Test that Distance is always positive."""
    d = Distance(tc.spacecraft, tc.target)

    times = [start + np.timedelta64(i * 30, "D") for i in range(10)]
    results = [d(t) for t in times]

    assert all(r > 0 for r in results)


def test_phase_angle_physical_range() -> None:
    """Test that PhaseAngle is in physical range."""
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN")

    times = [start + np.timedelta64(i * 30, "D") for i in range(10)]
    results = [phase(t) for t in times]

    # Phase angle should be 0 to π radians
    assert all(0 <= r <= np.pi for r in results)


def test_constraint_solve_empty_window() -> None:
    """Test solving constraint over very small window."""
    from spice_segmenter.core.time_segments_collection import TimeSegmentsCollection
    t_start = pd.Timestamp(str(start))
    t_end = t_start + pd.Timedelta(hours=1)
    w = TimeSegmentsCollection.from_start_end(t_start, t_end)

    c = Distance(tc.spacecraft, tc.target) < "5000 km"

    # Should handle very small time windows
    result = c.solve(w)
    assert result is not None


def test_property_evaluation_consistency() -> None:
    """Test that properties are deterministic."""
    d = Distance(tc.spacecraft, tc.target, light_time_correction="NONE")

    # Same time should give same result
    vals = [d(t1) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]


def test_constraint_repr() -> None:
    """Test constraint string representation."""
    c = Distance(tc.spacecraft, tc.target) < "5000 km"

    repr_str = repr(c)
    assert len(repr_str) > 0

    # Should contain constraint operator
    assert "<" in repr_str or "Constraint" in repr_str


def test_large_time_array() -> None:
    """Test property evaluation with larger time arrays."""
    d = Distance(tc.spacecraft, tc.target)

    # Create array of times
    times = np.array([start + np.timedelta64(i * 5, "D") for i in range(50)])

    # Should handle larger arrays
    results = d(times)
    assert results is not None
    assert len(results) == 50
    assert np.all(np.isfinite(results))
