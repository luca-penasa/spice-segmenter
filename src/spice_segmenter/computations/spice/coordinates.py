"""SPICE compute functions for coordinate/vector properties.

All functions receive a pre-converted float ET (seconds past J2000 TDB)
and the instantiated property object.  They mirror the logic that was
previously inlined as ``@vectorize __call__`` overrides in ``coordinates.py``.
"""

from __future__ import annotations

import numpy as np
import spiceypy
from spiceypy import NotFoundError

from ...properties.coordinates import (
    Boresight,
    BoresightIntersection,
    BoresightIntersects,
    CylindricalCoordinates,
    GeodeticCoordinates,
    LatitudinalCoordinates,
    PlanetographicCoordinates,
    RaDecCoordinates,
    SphericalCoordinates,
    SubObserverPoint,
    Vector,
)

# ---------------------------------------------------------------------------
# Vector  (spkpos)
# ---------------------------------------------------------------------------

def vector_scalar(prop: Vector, time_et: float) -> np.ndarray:
    return spiceypy.spkpos(
        prop.target.name, time_et, prop.frame, prop.abcorr, prop.origin.name,
    )[0]


def vector_vector(prop: Vector, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice
    positions, _ = cyice.spkpos_v(
        prop.target.name, times_et, str(prop.frame), prop.abcorr, prop.origin.name,
    )
    return positions  # (N, 3)


# ---------------------------------------------------------------------------
# SubObserverPoint  (subpnt — hardcodes "None" for abcorr, matching original)
# ---------------------------------------------------------------------------

def sub_observer_point_scalar(prop: SubObserverPoint, time_et: float) -> np.ndarray:
    return spiceypy.subpnt(
        str(prop.method),
        prop.target.name,
        time_et,
        prop.target.frame.name,
        "None",
        prop.origin.name,
    )[0]


def sub_observer_point_vector(prop: SubObserverPoint, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice
    spoints, _, _ = cyice.subpnt_v(
        str(prop.method),
        prop.target.name,
        times_et,
        prop.target.frame.name,
        "None",
        prop.origin.name,
    )
    return spoints  # (N, 3)


# ---------------------------------------------------------------------------
# Boresight  (pxform then apply to [0,0,1])
# ---------------------------------------------------------------------------

def boresight_scalar(prop: Boresight, time_et: float) -> np.ndarray:
    T = spiceypy.pxform(prop.instrument.frame, prop.frame, time_et)
    return T @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# BoresightIntersection  (sincpt)
# ---------------------------------------------------------------------------

def boresight_intersection_scalar(
    prop: BoresightIntersection, time_et: float,
) -> np.ndarray:
    # Compute boresight direction using the shared Boresight logic
    bsight = boresight_scalar(prop, time_et)
    try:
        return spiceypy.sincpt(
            "ELLIPSOID",
            prop.target.name,
            time_et,
            prop.target.frame.name,
            prop.abcorr,
            prop.instrument.name,
            prop.frame,
            bsight,
        )[0]
    except NotFoundError:
        return np.array([np.nan, np.nan, np.nan])


# ---------------------------------------------------------------------------
# BoresightIntersects  (bool – calls intersection engine fn)
# ---------------------------------------------------------------------------

def boresight_intersects_scalar(prop: BoresightIntersects, time_et: float) -> bool:
    from ...engines.evaluator import get_evaluator
    xyz = get_evaluator().evaluate_scalar(prop.intersection, time_et)
    return bool(~np.isnan(xyz).any())


# ---------------------------------------------------------------------------
# Coordinate transformations  (delegate vector computation to evaluator)
# ---------------------------------------------------------------------------

def _get_vec(prop, time_et: float) -> np.ndarray:
    from ...engines.evaluator import get_evaluator
    return get_evaluator().evaluate_scalar(prop.vector, time_et)


def latitudinal_scalar(prop: LatitudinalCoordinates, time_et: float) -> np.ndarray:
    value = _get_vec(prop, time_et)
    if np.isnan(value).any():
        return np.array([np.nan, np.nan, np.nan])
    return np.array(spiceypy.reclat(value))


def spherical_scalar(prop: SphericalCoordinates, time_et: float) -> np.ndarray:
    value = _get_vec(prop, time_et)
    if np.isnan(value).any():
        return np.array([np.nan, np.nan, np.nan])
    return np.array(spiceypy.recsph(value))


def cylindrical_scalar(prop: CylindricalCoordinates, time_et: float) -> np.ndarray:
    value = _get_vec(prop, time_et)
    if np.isnan(value).any():
        return np.array([np.nan, np.nan, np.nan])
    return np.array(spiceypy.reccyl(value))


def geodetic_scalar(prop: GeodeticCoordinates, time_et: float) -> np.ndarray:
    value = _get_vec(prop, time_et)
    if np.isnan(value).any():
        return np.array([np.nan, np.nan, np.nan])
    return np.array(spiceypy.recgeo(value))


def planetographic_scalar(prop: PlanetographicCoordinates, time_et: float) -> np.ndarray:
    value = _get_vec(prop, time_et)
    if np.isnan(value).any():
        return np.array([np.nan, np.nan, np.nan])
    return np.array(
        spiceypy.recpgr(
            prop.vector.target.name,
            value,
            prop.vector.target.re,
            prop.vector.target.f,
        ),
    )


def radec_scalar(prop: RaDecCoordinates, time_et: float) -> np.ndarray:
    return np.array(spiceypy.recrad(_get_vec(prop, time_et)))
