"""Coordinate transforms return angles in the units they declare.

Each transform (``as_latitudinal``, ``as_spherical``, ...) is checked against
the SPICE routine it wraps. They were registered with degree compute units
while SPICE returns radians, so every angle came out converted to radians a
second time (57x too small), and constraints on them were silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy

from spice_segmenter import SubObserverPoint, Vector
from spice_segmenter.io.dsl import parse
from tests import tour_config  # noqa: F401 — loads SPICE kernels as a side-effect

OBSERVER, TARGET, FRAME = "JUICE_JANUS", "JUPITER", "IAU_JUPITER"
METHOD = "INTERCEPT/ELLIPSOID"
TIMES = [
    "2032-03-01T00:00:00",
    "2032-11-13T18:00:00",
    "2033-06-15T06:00:00",
    "2033-11-22T12:00:00",
    "2034-04-10T00:00:00",
]


@pytest.fixture(scope="module")
def ets() -> np.ndarray:
    return np.array([spiceypy.str2et(t) for t in TIMES])


def _position(et: float, frame: str) -> np.ndarray:
    return spiceypy.spkpos(TARGET, et, frame, "NONE", OBSERVER)[0]


def _sub_point(et: float) -> np.ndarray:
    return spiceypy.subpnt(METHOD, TARGET, et, FRAME, "NONE", OBSERVER)[0]


def _shape() -> tuple[float, float]:
    radii = spiceypy.bodvrd(TARGET, "RADII", 3)[1]
    return radii[0], (radii[0] - radii[2]) / radii[0]


def _vector(frame: str = FRAME) -> Vector:
    return Vector(OBSERVER, TARGET, frame=frame)


def _sub_observer_point() -> SubObserverPoint:
    return SubObserverPoint(target=TARGET, origin=OBSERVER, frame=FRAME)


CASES = {
    "latitudinal": (
        lambda: _vector().as_latitudinal,
        lambda et: spiceypy.reclat(_position(et, FRAME)),
    ),
    "spherical": (
        lambda: _vector().as_spherical,
        lambda et: spiceypy.recsph(_position(et, FRAME)),
    ),
    "cylindrical": (
        lambda: _vector().as_cylindrical,
        lambda et: spiceypy.reccyl(_position(et, FRAME)),
    ),
    "radec": (
        lambda: _vector("J2000").as_radec,
        lambda et: spiceypy.recrad(_position(et, "J2000")),
    ),
    "geodetic": (
        lambda: _sub_observer_point().as_geodetic,
        lambda et: spiceypy.recgeo(_sub_point(et), *_shape()),
    ),
    "planetographic": (
        lambda: _sub_observer_point().as_planetographic,
        lambda et: spiceypy.recpgr(TARGET, _sub_point(et), *_shape()),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_transform_matches_spice(case: str, ets: np.ndarray) -> None:
    """SPICE's rec* routines give km and radians: exactly the declared units."""
    make_property, reference = CASES[case]
    prop = make_property()
    expected = np.array([reference(et) for et in ets])

    scalar = np.array([prop(float(et)) for et in ets])
    vector = np.asarray(prop(ets))

    np.testing.assert_allclose(scalar, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(vector, expected, rtol=1e-9, atol=1e-9)
    assert "radian" in {str(unit) for unit in prop.unit}


def test_latitude_as_degrees(ets: np.ndarray) -> None:
    latitude = _vector().as_latitudinal.latitude
    expected = np.degrees([spiceypy.reclat(_position(et, FRAME))[2] for et in ets])

    np.testing.assert_allclose(latitude.as_unit("deg")(ets), expected, rtol=1e-9)


@pytest.mark.parametrize("offset_deg", [-0.5, 0.5])
def test_latitude_constraint_uses_the_right_side_of_the_threshold(
    ets: np.ndarray,
    offset_deg: float,
) -> None:
    """``latitude < X deg`` decides like SPICE, just above and just below X."""
    latitude = _vector().as_latitudinal.latitude
    for et in ets:
        true_lat = np.degrees(spiceypy.reclat(_position(et, FRAME))[2])
        threshold = true_lat + offset_deg
        constraint = latitude < f"{threshold} deg"
        assert bool(constraint(float(et))) == bool(true_lat < threshold)


def test_ring_opening_angle_matches_sub_observer_latitude(ets: np.ndarray) -> None:
    """The ring opening angle B, taken as the latitude of the observer-to-Jupiter
    vector in Jupiter's frame, is minus the planetocentric latitude of the
    (intercept) sub-observer point: |B| agrees with ``sub_sc_latitude``."""
    opening_angle = _vector().as_latitudinal.latitude.as_unit("deg")
    sub_sc_latitude = parse(
        "sub_sc_latitude",
        context={"observer": OBSERVER, "target": TARGET},
    ).left

    np.testing.assert_allclose(opening_angle(ets), -sub_sc_latitude(ets), atol=1e-9)
