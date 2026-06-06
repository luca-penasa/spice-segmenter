"""SPICE compute functions for geometry properties.

Wraps the private ``_subpnt_*`` / ``_sincpt_*`` / ``_target_radec`` /
``_boresight_radec`` helpers that already live in
:mod:`~spice_segmenter.properties.geometry_properties`.

The helpers are imported directly so computation logic is written exactly once.
All boresight properties require an actual :class:`~planetary_coverage.SpiceInstrument`,
so those compute functions construct one from ``prop.observer.name``.
"""

from __future__ import annotations

import numpy as np
from planetary_coverage.spice import SpiceInstrument

from ...properties.geometry_properties import (
    # property classes
    BoresightAltitude,
    BoresightDec,
    BoresightGeoLatitude,
    BoresightGeoLongitude,
    BoresightIntersectionGeodetic,
    BoresightIntersectionLatitudinal,
    BoresightIntersectionRectangular,
    BoresightLatitude,
    BoresightLongitude,
    BoresightRA,
    BoresightRaDec,
    BoresightRadius,
    BoresightX,
    BoresightY,
    BoresightZ,
    SubObserverAltitude,
    SubObserverGeodetic,
    SubObserverGeoLatitude,
    SubObserverGeoLongitude,
    SubObserverLatitude,
    SubObserverLatitudinal,
    SubObserverLongitude,
    SubObserverRadius,
    SubObserverRectangular,
    SubObserverX,
    SubObserverY,
    SubObserverZ,
    TargetDec,
    TargetRA,
    TargetRaDec,
    # private helpers
    _boresight_radec,
    _sincpt_geodetic,
    _sincpt_latitudinal,
    _sincpt_xyz,
    _subpnt_geodetic,
    _subpnt_geodetic_v,
    _subpnt_latitudinal,
    _subpnt_latitudinal_v,
    _subpnt_xyz,
    _subpnt_xyz_v,
    _target_radec,
)

# ===========================================================================
# Sub-observer point — Latitudinal
# ===========================================================================

def sub_sc_latitudinal_scalar(prop: SubObserverLatitudinal, time_et: float) -> np.ndarray:
    return _subpnt_latitudinal(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )


def sub_sc_latitudinal_vector(
    prop: SubObserverLatitudinal, times_et: np.ndarray,
) -> np.ndarray:
    r, lon, lat = _subpnt_latitudinal_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return np.stack([r, lon, lat], axis=-1)


# --- scalar components ---

def sub_sc_radius_scalar(prop: SubObserverRadius, time_et: float) -> float:
    return _subpnt_latitudinal(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[0]


def sub_sc_radius_vector(prop: SubObserverRadius, times_et: np.ndarray) -> np.ndarray:
    r, _, _ = _subpnt_latitudinal_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return r


def sub_sc_longitude_scalar(prop: SubObserverLongitude, time_et: float) -> float:
    return _subpnt_latitudinal(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[1]


def sub_sc_longitude_vector(
    prop: SubObserverLongitude, times_et: np.ndarray,
) -> np.ndarray:
    _, lon, _ = _subpnt_latitudinal_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return lon


def sub_sc_latitude_scalar(prop: SubObserverLatitude, time_et: float) -> float:
    return _subpnt_latitudinal(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[2]


def sub_sc_latitude_vector(
    prop: SubObserverLatitude, times_et: np.ndarray,
) -> np.ndarray:
    _, _, lat = _subpnt_latitudinal_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return lat


# ===========================================================================
# Sub-observer point — Geodetic
# ===========================================================================

def sub_sc_geodetic_scalar(prop: SubObserverGeodetic, time_et: float) -> np.ndarray:
    return _subpnt_geodetic(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )


def sub_sc_geodetic_vector(
    prop: SubObserverGeodetic, times_et: np.ndarray,
) -> np.ndarray:
    lon, lat, alt = _subpnt_geodetic_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return np.stack([lon, lat, alt], axis=-1)


# --- scalar components ---

def sub_sc_geo_longitude_scalar(prop: SubObserverGeoLongitude, time_et: float) -> float:
    return _subpnt_geodetic(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[0]


def sub_sc_geo_longitude_vector(
    prop: SubObserverGeoLongitude, times_et: np.ndarray,
) -> np.ndarray:
    lon, _, _ = _subpnt_geodetic_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return lon


def sub_sc_geo_latitude_scalar(prop: SubObserverGeoLatitude, time_et: float) -> float:
    return _subpnt_geodetic(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[1]


def sub_sc_geo_latitude_vector(
    prop: SubObserverGeoLatitude, times_et: np.ndarray,
) -> np.ndarray:
    _, lat, _ = _subpnt_geodetic_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return lat


def sub_sc_altitude_scalar(prop: SubObserverAltitude, time_et: float) -> float:
    return _subpnt_geodetic(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[2]


def sub_sc_altitude_vector(
    prop: SubObserverAltitude, times_et: np.ndarray,
) -> np.ndarray:
    _, _, alt = _subpnt_geodetic_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )
    return alt


# ===========================================================================
# Sub-observer point — Rectangular (body-fixed XYZ)
# ===========================================================================

def sub_sc_rectangular_scalar(
    prop: SubObserverRectangular, time_et: float,
) -> np.ndarray:
    return _subpnt_xyz(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )


def sub_sc_rectangular_vector(
    prop: SubObserverRectangular, times_et: np.ndarray,
) -> np.ndarray:
    return _subpnt_xyz_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )


def sub_sc_x_scalar(prop: SubObserverX, time_et: float) -> float:
    return _subpnt_xyz(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[0]


def sub_sc_x_vector(prop: SubObserverX, times_et: np.ndarray) -> np.ndarray:
    return _subpnt_xyz_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[:, 0]


def sub_sc_y_scalar(prop: SubObserverY, time_et: float) -> float:
    return _subpnt_xyz(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[1]


def sub_sc_y_vector(prop: SubObserverY, times_et: np.ndarray) -> np.ndarray:
    return _subpnt_xyz_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[:, 1]


def sub_sc_z_scalar(prop: SubObserverZ, time_et: float) -> float:
    return _subpnt_xyz(
        time_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[2]


def sub_sc_z_vector(prop: SubObserverZ, times_et: np.ndarray) -> np.ndarray:
    return _subpnt_xyz_v(
        times_et, prop.target, prop.observer, prop.method, prop.light_time_correction,
    )[:, 2]


# ===========================================================================
# Boresight intersection — Latitudinal
# ===========================================================================

def _inst(prop) -> SpiceInstrument:
    """Extract SpiceInstrument from a TargetedProperty observer field."""
    return SpiceInstrument(prop.observer.name)


def boresight_latitudinal_scalar(
    prop: BoresightIntersectionLatitudinal, time_et: float,
) -> np.ndarray:
    return _sincpt_latitudinal(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )


def boresight_latitudinal_vector(
    prop: BoresightIntersectionLatitudinal, times_et: np.ndarray,
) -> np.ndarray:
    inst = _inst(prop)
    return np.array([
        _sincpt_latitudinal(t, prop.target, inst, prop.light_time_correction)
        for t in times_et
    ])


def boresight_latitude_scalar(prop: BoresightLatitude, time_et: float) -> float:
    return _sincpt_latitudinal(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[2]


def boresight_longitude_scalar(prop: BoresightLongitude, time_et: float) -> float:
    return _sincpt_latitudinal(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[1]


def boresight_radius_scalar(prop: BoresightRadius, time_et: float) -> float:
    return _sincpt_latitudinal(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[0]


# ===========================================================================
# Boresight intersection — Geodetic
# ===========================================================================

def boresight_geodetic_scalar(
    prop: BoresightIntersectionGeodetic, time_et: float,
) -> np.ndarray:
    return _sincpt_geodetic(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )


def boresight_geodetic_vector(
    prop: BoresightIntersectionGeodetic, times_et: np.ndarray,
) -> np.ndarray:
    inst = _inst(prop)
    return np.array([
        _sincpt_geodetic(t, prop.target, inst, prop.light_time_correction)
        for t in times_et
    ])


def boresight_geo_latitude_scalar(prop: BoresightGeoLatitude, time_et: float) -> float:
    return _sincpt_geodetic(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[1]


def boresight_geo_longitude_scalar(
    prop: BoresightGeoLongitude, time_et: float,
) -> float:
    return _sincpt_geodetic(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[0]


def boresight_altitude_scalar(prop: BoresightAltitude, time_et: float) -> float:
    return _sincpt_geodetic(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )[2]


# ===========================================================================
# Boresight intersection — Rectangular (body-fixed XYZ)
# ===========================================================================

def boresight_rectangular_scalar(
    prop: BoresightIntersectionRectangular, time_et: float,
) -> np.ndarray:
    return _sincpt_xyz(
        time_et, prop.target, _inst(prop), prop.light_time_correction,
    )


def boresight_rectangular_vector(
    prop: BoresightIntersectionRectangular, times_et: np.ndarray,
) -> np.ndarray:
    inst = _inst(prop)
    return np.array([
        _sincpt_xyz(t, prop.target, inst, prop.light_time_correction)
        for t in times_et
    ])


def boresight_x_scalar(prop: BoresightX, time_et: float) -> float:
    return _sincpt_xyz(time_et, prop.target, _inst(prop), prop.light_time_correction)[0]


def boresight_y_scalar(prop: BoresightY, time_et: float) -> float:
    return _sincpt_xyz(time_et, prop.target, _inst(prop), prop.light_time_correction)[1]


def boresight_z_scalar(prop: BoresightZ, time_et: float) -> float:
    return _sincpt_xyz(time_et, prop.target, _inst(prop), prop.light_time_correction)[2]


# ===========================================================================
# Target RA / Dec
# ===========================================================================

def target_radec_scalar(prop: TargetRaDec, time_et: float) -> np.ndarray:
    return _target_radec(
        time_et, prop.target, prop.observer, prop.light_time_correction,
    )


def target_radec_vector(prop: TargetRaDec, times_et: np.ndarray) -> np.ndarray:
    return np.array([
        _target_radec(t, prop.target, prop.observer, prop.light_time_correction)
        for t in times_et
    ])


def target_ra_scalar(prop: TargetRA, time_et: float) -> float:
    return _target_radec(
        time_et, prop.target, prop.observer, prop.light_time_correction,
    )[0]


def target_dec_scalar(prop: TargetDec, time_et: float) -> float:
    return _target_radec(
        time_et, prop.target, prop.observer, prop.light_time_correction,
    )[1]


# ===========================================================================
# Boresight RA / Dec
# ===========================================================================

def boresight_radec_scalar(prop: BoresightRaDec, time_et: float) -> np.ndarray:
    return _boresight_radec(
        time_et, _inst(prop), prop.light_time_correction,
    )


def boresight_radec_vector(prop: BoresightRaDec, times_et: np.ndarray) -> np.ndarray:
    inst = _inst(prop)
    return np.array([
        _boresight_radec(t, inst, prop.light_time_correction)
        for t in times_et
    ])


def boresight_ra_scalar(prop: BoresightRA, time_et: float) -> float:
    return _boresight_radec(time_et, _inst(prop), prop.light_time_correction)[0]


def boresight_dec_scalar(prop: BoresightDec, time_et: float) -> float:
    return _boresight_radec(time_et, _inst(prop), prop.light_time_correction)[1]


# ---------------------------------------------------------------------------
# Sub-observer illumination angle scalars
# ---------------------------------------------------------------------------

from ...properties.geometry_properties import (  # noqa: E402
    SubObserverEmissionAngle,
    SubObserverIncidenceAngle,
    SubObserverPhaseAngleLocal,
    _sub_observer_illum,
)


def sub_observer_incidence_angle_scalar(
    prop: SubObserverIncidenceAngle, time_et: float,
) -> float:
    return float(_sub_observer_illum(time_et, prop.target, prop.observer, prop.light_time_correction)[0])


def sub_observer_emission_angle_scalar(
    prop: SubObserverEmissionAngle, time_et: float,
) -> float:
    return float(_sub_observer_illum(time_et, prop.target, prop.observer, prop.light_time_correction)[1])


def sub_observer_phase_angle_local_scalar(
    prop: SubObserverPhaseAngleLocal, time_et: float,
) -> float:
    return float(_sub_observer_illum(time_et, prop.target, prop.observer, prop.light_time_correction)[2])
