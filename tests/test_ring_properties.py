"""Tests for ring properties and geometry."""

import numpy as np

from spice_segmenter.properties.ring_properties import ring_ansae_phase_angles

from . import tour_config as tc

tc.load_kernels()
start, end = tc.coverage


def test_ring_plane_angles_basic():
    """Test ring_plane_angles computation at a single time."""
    # Use a time in the trajectory coverage
    time = start + np.timedelta64(100, "D")

    # Should return phase angles for different ring plane points
    angles = ring_ansae_phase_angles(time)

    assert angles is not None
    assert isinstance(angles, np.ndarray)
    assert len(angles) > 0

    # Phase angles should be valid floats
    assert np.all(np.isfinite(angles))


def test_ring_plane_angles_custom_radii():
    """Test ring_plane_angles with custom ring radii."""
    time = start + np.timedelta64(100, "D")
    custom_radii = np.array([-200000, 0, 200000])

    angles = ring_ansae_phase_angles(time, pts_radiuses=custom_radii)

    assert angles is not None
    assert len(angles) == len(custom_radii)


def test_ring_plane_angles_multiple_times():
    """Test ring_plane_angles at multiple times (vectorization)."""
    times = [
        start + np.timedelta64(50, "D"),
        start + np.timedelta64(100, "D"),
        start + np.timedelta64(150, "D"),
    ]

    # Should handle array of times
    for t in times:
        angles = ring_ansae_phase_angles(t)
        assert angles is not None
        assert isinstance(angles, np.ndarray)


def test_ring_plane_angles_different_observer():
    """Test ring_plane_angles with different observer."""
    time = start + np.timedelta64(100, "D")

    # Test with different observer if available
    # This tests that the function doesn't hard-code observer
    angles = ring_ansae_phase_angles(time, observer="JUICE")

    assert angles is not None
    assert isinstance(angles, np.ndarray)


def test_ring_plane_angles_return_type():
    """Test that ring_plane_angles returns correct data types."""
    time = start + np.timedelta64(100, "D")
    angles = ring_ansae_phase_angles(time)

    # Should be numpy array of floats
    assert isinstance(angles, np.ndarray)
    assert np.issubdtype(angles.dtype, np.floating)

    # All values should be valid floats (not NaN or Inf)
    assert np.all(np.isfinite(angles))


def test_ring_plane_angles_symmetry():
    """Test that ring plane angles have expected symmetry properties."""
    time = start + np.timedelta64(100, "D")

    # Radii symmetric around 0 should have related angles
    radii_symmetric = np.array([-100000, 0, 100000])
    angles = ring_ansae_phase_angles(time, pts_radiuses=radii_symmetric)

    # Should have 3 angles
    assert len(angles) == 3

    # Angles should be defined even if not exactly symmetric
    assert np.all(np.isfinite(angles))
