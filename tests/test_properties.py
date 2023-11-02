import numpy as np
import pint
from pytest import approx

from spice_segmenter.decorators import declare
from spice_segmenter.ops import Inverted
from spice_segmenter.trajectory_properties import (
    Constant,
    Constraint,
    Distance,
    PhaseAngle,
    Property,
    UnitAdaptor,
)
from spice_segmenter.types import TIMES_TYPES

from . import TourConfig

tc = TourConfig()
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
        np.rad2deg(ph1), abs=1e-2
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
        c.right, Constant
    )  # it should translate a 1 to a constant by itself.

    c2 = p == 1
    assert isinstance(
        c2.right, Constant
    )  # it should translate a 1 to a constant by itself.

    assert c("2020-01-01") == True
    p_ = Inverted(c)
    value = p_("2020-01-01")
    assert value == False
