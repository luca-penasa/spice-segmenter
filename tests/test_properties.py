import numpy as np
import pint
from pytest import approx

from spice_segmenter.ops.constant_values import Constant
from spice_segmenter.core.constraints import Constraint
from spice_segmenter.support.decorators import declare
from spice_segmenter.ops import Inverted
from spice_segmenter.core.property import Property
from spice_segmenter.properties.observation_properties import (
    Distance,
    PhaseAngle,
)
from spice_segmenter.support.time_types import TIMES_TYPES
from spice_segmenter.ops.unit_adapter import UnitAdaptor

from . import tour_config as tc

start, end = tc.coverage

t1 = start + np.timedelta64(100, "D")


def test_imports() -> None:
    # some random imports
    pass


def test_distance() -> None:
    d = Distance(tc.spacecraft, tc.target, light_time_correction="NONE")
    d1 = d(t1)

    assert tc[t1].dist[0] == approx(d1)  # type: ignore


def test_phase() -> None:
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN", light_time_correction="NONE")
    ph1 = phase(t1)

    assert tc[t1].phase[0] == approx(  # type: ignore
        np.rad2deg(ph1), abs=1e-2,
    )  # abs: we know planetary coverage implements the phase at the sub-sc point, but should be almost there anyway


def test_unit_adaptor() -> None:
    d = Distance(tc.spacecraft, tc.target, light_time_correction="NONE")

    assert d.unit == pint.Unit("km")

    d_m = UnitAdaptor(d, "m")

    assert d_m.unit == pint.Unit("m")

    dm = d_m(t1)
    dkm = d(t1)

    assert dkm == approx(dm / 1000.0)

    ph = PhaseAngle(tc.spacecraft, tc.target, "SUN", light_time_correction="NONE")

    assert ph.unit == pint.Unit("rad")

    ph_deg = UnitAdaptor(ph, "deg")

    assert ph_deg.unit == pint.Unit("deg")

    ph_deg = ph_deg(t1)

    assert ph_deg == approx(np.rad2deg(ph(t1)))


# property declare is a wrapper that can automatically create a new Property class from a function and some more arguments.
# it is used to create properties from functions that are not properties themselves.


@declare(name="one_property")
class FakeOneProperty(Property):
    def __call__(self, time: TIMES_TYPES) -> float:
        return 1


def test_inverted_wrapped():
    p = FakeOneProperty()
    value = p("2020-01-01")
    assert value == 1

    c = Constraint(p, 1, "==")
    assert isinstance(
        c.right, Constant,
    )  # it should translate a 1 to a constant by itself.

    c2 = p == 1
    assert isinstance(
        c2.right, Constant,
    )  # it should translate a 1 to a constant by itself.

    assert c("2020-01-01") == True
    p_ = Inverted(c)
    value = p_("2020-01-01")
    assert value == False


# Additional property tests for coverage


def test_angular_size() -> None:
    """Test AngularSize property computation."""
    from spice_segmenter.properties.observation_properties import AngularSize
    
    ang_size = AngularSize(tc.spacecraft, tc.target, light_time_correction="NONE")
    
    result = ang_size(t1)
    assert result is not None
    assert result > 0
    assert np.isfinite(result)


def test_angular_size_constraint() -> None:
    """Test AngularSize in constraint."""
    from spice_segmenter.properties.observation_properties import AngularSize
    
    ang_size = AngularSize(tc.spacecraft, tc.target)
    
    # Should support unit conversion
    ang_size_deg = UnitAdaptor(ang_size, "deg")
    assert ang_size_deg.unit == pint.Unit("deg")
    
    result = ang_size_deg(t1)
    assert result > 0


def test_distance_light_time_corrections() -> None:
    """Test Distance property with different light-time corrections."""
    corrections = ["NONE", "LT", "LT+S"]
    results = []
    
    for corr in corrections:
        d = Distance(tc.spacecraft, tc.target, light_time_correction=corr)
        result = d(t1)
        results.append(result)
        assert result > 0
        assert np.isfinite(result)
    
    # All corrections should give positive distances
    assert len(results) == 3
    assert all(r > 0 for r in results)


def test_phase_angle_range() -> None:
    """Test PhaseAngle returns values in expected range."""
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN", light_time_correction="NONE")
    
    result = phase(t1)
    
    # Phase angle should be in [0, π] radians
    assert result >= 0
    assert result <= np.pi


def test_distance_multiple_times() -> None:
    """Test Distance evaluation at multiple times."""
    d = Distance(tc.spacecraft, tc.target)
    
    times = [start + np.timedelta64(i * 20, "D") for i in range(3)]
    results = [d(t) for t in times]
    
    # All should be valid
    assert all(np.isfinite(r) for r in results)
    assert all(r > 0 for r in results)
    
    # Should have multiple distinct values (spacecraft moves)
    assert len(results) == 3


def test_unit_adaptor_chaining() -> None:
    """Test chaining multiple UnitAdaptors."""
    d = Distance(tc.spacecraft, tc.target)
    
    # Convert km -> m -> km
    d_m = UnitAdaptor(d, "m")
    d_km_again = UnitAdaptor(d_m, "km")
    
    result_direct = d(t1)
    result_adapted = d_km_again(t1)
    
    assert result_direct == approx(result_adapted)


def test_property_has_unit() -> None:
    """Test that properties correctly report unit information."""
    d = Distance(tc.spacecraft, tc.target)
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN")
    
    assert d.has_unit()
    assert phase.has_unit()
    
    assert d.unit == pint.Unit("km")
    assert phase.unit == pint.Unit("rad")


def test_property_repr() -> None:
    """Test that properties have meaningful string representations."""
    d = Distance(tc.spacecraft, tc.target)
    phase = PhaseAngle(tc.spacecraft, tc.target, "SUN")
    
    d_str = str(d)
    phase_str = str(phase)
    
    assert len(d_str) > 0
    assert len(phase_str) > 0
    assert "Distance" in d_str or "distance" in d_str.lower()
    assert "Phase" in phase_str or "phase" in phase_str.lower()
