import numpy as np
import pint
from pytest import approx

from spice_segmenter.trajectory_properties import Distance, PhaseAngle, UnitAdaptor

from . import TourConfig

tc = TourConfig()
start, end = tc.coverage

t1 = start + np.timedelta64(100, "D")


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
