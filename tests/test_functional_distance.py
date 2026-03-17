"""Tests for function-based Distance property."""

import pytest
import numpy as np
from spice_segmenter.properties.functional import Distance
from spice_segmenter.properties.observation_properties import Distance as ClassDistance
from spice_segmenter.core.constraints import Constraint
from spice_segmenter.support.spice_utilities import et
from . import tour_config as tc

tc.load_kernels()


def test_functional_distance_creation():
    """Test creating functional Distance property."""
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    assert hasattr(dist, 'observer')
    assert hasattr(dist, 'target')
    assert dist.name == "distance"
    assert str(dist.unit) == "kilometer"


def test_functional_distance_computation():
    """Test computing distance with functional property."""
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    start, end = tc.coverage
    result = dist(start)
    
    assert isinstance(result, (float, np.ndarray))
    assert result > 0  # Distance should be positive


def test_functional_distance_constraint():
    """Test creating constraints with functional Distance."""
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    constraint = dist < "100000 km"
    
    assert isinstance(constraint, Constraint)
    assert constraint.operator == "<"


def test_functional_vs_class_equivalence():
    """Test that functional and class-based Distance give same results."""
    start, end = tc.coverage
    
    # Create both versions
    dist_func = Distance("JUICE_JANUS", "GANYMEDE")
    dist_class = ClassDistance("JUICE_JANUS", "GANYMEDE")
    
    # Compare results
    result_func = dist_func(start)
    result_class = dist_class(start)
    
    assert np.allclose(result_func, result_class)


def test_functional_distance_with_correction():
    """Test functional Distance with light time correction."""
    dist1 = Distance("JUICE_JANUS", "GANYMEDE", light_time_correction="NONE")
    dist2 = Distance("JUICE_JANUS", "GANYMEDE", light_time_correction="LT")
    
    start, end = tc.coverage
    
    result1 = dist1(start)
    result2 = dist2(start)
    
    # Results should be slightly different
    assert result1 != result2
    # But both should be positive and in reasonable range
    assert result1 > 0
    assert result2 > 0


def test_functional_distance_vectorized():
    """Test that functional Distance handles arrays."""
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    start, end = tc.coverage
    # Use ET times (floats) for array
    start_et = et(start)
    times = np.linspace(start_et, start_et + 86400, 10)
    results = dist(times)
    
    assert isinstance(results, np.ndarray)
    assert len(results) == 10
    assert np.all(results > 0)


def test_functional_distance_metadata():
    """Test that functional Distance has proper metadata."""
    assert hasattr(Distance, '_metadata')
    assert hasattr(Distance, '_compute_fn')
    
    metadata = Distance._metadata
    assert metadata.name == "distance"
    assert 'observer' in metadata.parameter_names
    assert 'target' in metadata.parameter_names
    assert 'light_time_correction' in metadata.parameter_defaults
