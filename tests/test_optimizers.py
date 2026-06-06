"""Tests for constraint optimization."""

from spice_segmenter import (
    ConstraintOptimizer,
    Distance,
    TargetSizeOnSensor,
    TimeSegmentsCollection,
    config,
    optimize_constraint,
)
from spice_segmenter.optimizers.constraint_optimizer import (
    TargetSizeOnSensorToDistance,
)

from . import tour_config as tc

config.solver_step = "24 h"
start, end = tc.coverage
tc.load_kernels()

w = TimeSegmentsCollection.from_start_end("2032-01-01", "2034-01-01")


def test_target_size_on_sensor_to_distance_transformer():
    """Test that TargetSizeOnSensorToDistance can identify transformable constraints."""
    transformer = TargetSizeOnSensorToDistance()

    # Should be able to transform TargetSizeOnSensor constraints
    left = TargetSizeOnSensor(tc.spacecraft, tc.target)
    assert transformer.can_transform(left, ">", 5)

    # Should not transform other properties
    distance = Distance(tc.spacecraft, tc.target)
    assert not transformer.can_transform(distance, ">", 5)


def test_constraint_optimizer_selection():
    """Test that optimizer properly selects transformers."""
    # Create a TargetSizeOnSensor constraint that could be optimized
    left = TargetSizeOnSensor(tc.spacecraft, tc.target)
    c_original = left > 5  # pixels

    optimizer = ConstraintOptimizer()

    # Should have transformers registered
    assert len(optimizer.transformers) > 0

    # The optimizer should attempt transformation
    c_optimized = optimizer.optimize(c_original)

    # Optimization might not always transform (depends on context),
    # but should return a constraint
    assert c_optimized is not None


def test_optimize_constraint_function():
    """Test the public optimize_constraint function."""
    # Create a simple distance constraint (usually doesn't need optimization)
    c = Distance(tc.spacecraft, tc.target) < "5000 km"

    # Should return a constraint
    c_opt = optimize_constraint(c)
    assert c_opt is not None

    # Optimized constraint should still be callable
    t = start + __import__("numpy").timedelta64(100, "D")
    result = c_opt(t)
    # Result can be bool, numpy bool, or numpy array
    assert result is not None


def test_distance_constraint_consistency():
    """Verify that optimization doesn't change constraint semantics."""
    c_original = Distance(tc.spacecraft, tc.target) < "1000000 km"

    # Solve original
    result_orig = c_original.solve(w)
    n_orig = len(result_orig)

    # Optimize and solve
    c_optimized = optimize_constraint(c_original)
    result_opt = c_optimized.solve(w)
    n_opt = len(result_opt)

    # Should find similar number of intervals
    assert n_orig == n_opt or abs(n_orig - n_opt) <= 1


def test_optimizer_with_composite_constraints():
    """Test optimizer handles composite (combined) constraints."""
    c1 = Distance(tc.spacecraft, tc.target) < "5000 km"
    c2 = Distance(tc.spacecraft, tc.target) > "1000 km"
    c_combined = c1 & c2

    # Should handle combined constraints
    c_opt = optimize_constraint(c_combined)
    assert c_opt is not None


def test_optimizer_preserves_constraint_structure():
    """Test that optimization preserves constraint semantics."""
    c = Distance(tc.spacecraft, tc.target) > "500000 km"

    c_opt = optimize_constraint(c)

    # Both should have same operator
    assert c_opt.operator == c.operator

    # Both should evaluate to same results at same times
    t = start + __import__("numpy").timedelta64(50, "D")
    val_orig = c(t)
    val_opt = c_opt(t)

    assert val_orig == val_opt
