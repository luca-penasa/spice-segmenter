"""Concrete registered geometry properties derived from SPICE position &
surface-point computations.

Architecture
------------
Each physical *vector* quantity (sub-observer point in latitudinal coordinates,
boresight intersection geodetic coordinates, …) has:

1. A **vector parent class** — registered under its ``_name``, returns a
   numpy array of components, inherits ``TargetedProperty`` so it is
   auto-instantiable from context (observer / target).

2. **Scalar component classes** — one per meaningful scalar derived from the
   vector.  Each is independently registered, makes the same underlying SPICE
   call via a shared private helper, and returns the single component of
   interest.  No chaining through the parent at runtime.

3. **ComponentSelector @properties on the parent** — unchanged navigation API
   for composition chains and constraint DSL dot-notation
   (e.g. ``SubObserverLatitudinal(...).latitude``).

Private helpers (prefixed ``_``) contain the raw SPICE call so it is written
exactly once and both the vector parent and every scalar component share it
without re-vectorising.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pint
import spiceypy
from attrs import define, field
from planetary_coverage import et
from planetary_coverage.spice import SpiceInstrument
from spiceypy import NotFoundError

from ..core.property import Property, PropertyTypes
from ..properties.component_selector import ComponentSelector
from ..properties.observation_properties import TargetedProperty

# ---------------------------------------------------------------------------
# Private SPICE helpers (non-vectorised, scalar-time)
# ---------------------------------------------------------------------------

_SUBPNT_METHOD = "INTERCEPT/ELLIPSOID"


def _subpnt_xyz(time, target, observer, method: str, abcorr: str) -> np.ndarray:
    """Return sub-observer surface point (x, y, z) in body-fixed frame."""
    return spiceypy.subpnt(
        method,
        target.name,
        et(time),
        target.frame.name,
        abcorr,
        observer.name,
    )[0]


def _subpnt_latitudinal(time, target, observer, method: str, abcorr: str) -> np.ndarray:
    """Return (radius_km, longitude_deg, latitude_deg) for sub-observer point."""
    xyz = _subpnt_xyz(time, target, observer, method, abcorr)
    r, lon, lat = spiceypy.reclat(xyz)
    return np.array([r, np.rad2deg(lon), np.rad2deg(lat)])


def _subpnt_geodetic(time, target, observer, method: str, abcorr: str) -> np.ndarray:
    """Return (longitude_deg, latitude_deg, altitude_km) for sub-observer point."""
    xyz = _subpnt_xyz(time, target, observer, method, abcorr)
    lon, lat, alt = spiceypy.recgeo(xyz, target.re, target.f)
    return np.array([np.rad2deg(lon), np.rad2deg(lat), alt])


# ---------------------------------------------------------------------------
# Vectorized SPICE helpers (cyice _v — C-level batch operations)
# These accept an np.ndarray of float64 ET values already converted from
# TIMES_TYPES by Property.__call__ and return plain numpy arrays.
# ---------------------------------------------------------------------------

def _subpnt_latitudinal_v(
    times_et: np.ndarray, target, observer, method: str, abcorr: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (radius_km, longitude_deg, latitude_deg) for N ET values."""
    from spiceypy import cyice
    spoints, _, _ = cyice.subpnt_v(
        method, target.name, times_et, target.frame.name, abcorr, observer.name,
    )
    # reclat_v returns (N, 3) where columns are (radius, lon_rad, lat_rad)
    latitudinal = cyice.reclat_v(spoints)
    return latitudinal[:, 0], np.rad2deg(latitudinal[:, 1]), np.rad2deg(latitudinal[:, 2])


def _subpnt_geodetic_v(
    times_et: np.ndarray, target, observer, method: str, abcorr: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (longitude_deg, latitude_deg, altitude_km) for N ET values."""
    from spiceypy import cyice
    spoints, _, _ = cyice.subpnt_v(
        method, target.name, times_et, target.frame.name, abcorr, observer.name,
    )
    # recgeo_v returns (N, 3) where columns are (lon_rad, lat_rad, alt_km)
    geodetic = cyice.recgeo_v(spoints, target.re, target.f)
    return np.rad2deg(geodetic[:, 0]), np.rad2deg(geodetic[:, 1]), geodetic[:, 2]


def _subpnt_xyz_v(
    times_et: np.ndarray, target, observer, method: str, abcorr: str,
) -> np.ndarray:
    """Vectorized body-fixed XYZ (N, 3) for N ET values."""
    from spiceypy import cyice
    spoints, _, _ = cyice.subpnt_v(
        method, target.name, times_et, target.frame.name, abcorr, observer.name,
    )
    return spoints


_NAN3 = np.array([np.nan, np.nan, np.nan])


def _sincpt_xyz(time, target, observer: SpiceInstrument, abcorr: str) -> np.ndarray:
    """Return boresight-surface intersection (x, y, z) in body-fixed frame.

    Returns ``[NaN, NaN, NaN]`` when the boresight misses the body.
    """
    try:
        return spiceypy.sincpt(
            "ELLIPSOID",
            target.name,
            et(time),
            target.frame.name,
            abcorr,
            observer.name,
            str(observer.frame),
            np.array([0.0, 0.0, 1.0]),
        )[0]
    except NotFoundError:
        return _NAN3.copy()


def _sincpt_latitudinal(time, target, observer: SpiceInstrument, abcorr: str) -> np.ndarray:
    """Return (radius_km, longitude_deg, latitude_deg) for boresight intersection."""
    xyz = _sincpt_xyz(time, target, observer, abcorr)
    if np.isnan(xyz).any():
        return _NAN3.copy()
    r, lon, lat = spiceypy.reclat(xyz)
    return np.array([r, np.rad2deg(lon), np.rad2deg(lat)])


def _sincpt_geodetic(time, target, observer: SpiceInstrument, abcorr: str) -> np.ndarray:
    """Return (longitude_deg, latitude_deg, altitude_km) for boresight intersection."""
    xyz = _sincpt_xyz(time, target, observer, abcorr)
    if np.isnan(xyz).any():
        return _NAN3.copy()
    lon, lat, alt = spiceypy.recgeo(xyz, target.re, target.f)
    return np.array([np.rad2deg(lon), np.rad2deg(lat), alt])


def _target_radec(time, target, observer, abcorr: str) -> np.ndarray:
    """Return (right_ascension_deg, declination_deg) of target from observer in J2000."""
    pos = spiceypy.spkpos(target.name, et(time), "J2000", abcorr, observer.name)[0]
    _, ra, dec = spiceypy.recrad(pos)
    return np.array([np.rad2deg(ra), np.rad2deg(dec)])


def _boresight_radec(time, observer: SpiceInstrument, abcorr: str) -> np.ndarray:
    """Return (right_ascension_deg, declination_deg) of instrument boresight in J2000."""
    T = spiceypy.pxform(str(observer.frame), "J2000", et(time))
    pos = T @ np.array([0.0, 0.0, 1.0])
    _, ra, dec = spiceypy.recrad(pos)
    return np.array([np.rad2deg(ra), np.rad2deg(dec)])


# ---------------------------------------------------------------------------
# Sub-observer point — Latitudinal coordinates
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class SubObserverLatitudinal(TargetedProperty):
    """Sub-observer point expressed in latitudinal (radius, longitude, latitude) coordinates.

    Components (each independently registered as a scalar property):

    * ``sub_sc_radius`` — distance from body centre to sub-observer
      point (km)
    * ``sub_sc_longitude`` — longitude of the sub-observer point (deg,
      planetocentric, positive east)
    * ``sub_sc_latitude`` — latitude of the sub-observer point (deg,
      planetocentric)
    """

    _name: ClassVar[str] = "sub_sc_latitudinal"
    _unit: ClassVar[list] = [pint.Unit("km"), pint.Unit("deg"), pint.Unit("deg")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"Sub-observer ({self.observer}) latitudinal coordinates on "
            f"{self.target} [radius, lon, lat]"
        )

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "sub_sc_radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "sub_sc_longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 2, "sub_sc_latitude")


@define(repr=False, order=False, eq=False)
class SubObserverRadius(TargetedProperty):
    """Distance from body centre to the sub-observer point (km)."""

    _name: ClassVar[str] = "sub_sc_radius"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer radius on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverLongitude(TargetedProperty):
    """Planetocentric longitude of the sub-observer point (deg, positive east)."""

    _name: ClassVar[str] = "sub_sc_longitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer longitude on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverLatitude(TargetedProperty):
    """Planetocentric latitude of the sub-observer point (deg)."""

    _name: ClassVar[str] = "sub_sc_latitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer latitude on {self.target} from {self.observer}"


# ---------------------------------------------------------------------------
# Sub-observer point — Geodetic coordinates
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class SubObserverGeodetic(TargetedProperty):
    """Sub-observer point expressed in geodetic (longitude, latitude, altitude) coordinates.

    Components:

    * ``sub_sc_geo_longitude`` — geodetic longitude (deg)
    * ``sub_sc_geo_latitude`` — geodetic latitude (deg)
    * ``sub_sc_altitude`` — altitude above reference ellipsoid (km)
    """

    _name: ClassVar[str] = "sub_sc_geodetic"
    _unit: ClassVar[list] = [pint.Unit("deg"), pint.Unit("deg"), pint.Unit("km")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"Sub-observer ({self.observer}) geodetic coordinates on "
            f"{self.target} [lon, lat, alt]"
        )

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "sub_sc_geo_longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "sub_sc_geo_latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "sub_sc_altitude")


@define(repr=False, order=False, eq=False)
class SubObserverGeoLongitude(TargetedProperty):
    """Geodetic longitude of the sub-observer point (deg)."""

    _name: ClassVar[str] = "sub_sc_geo_longitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer geodetic longitude on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverGeoLatitude(TargetedProperty):
    """Geodetic latitude of the sub-observer point (deg)."""

    _name: ClassVar[str] = "sub_sc_geo_latitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer geodetic latitude on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverAltitude(TargetedProperty):
    """Geodetic altitude of the sub-observer point above the reference ellipsoid (km)."""

    _name: ClassVar[str] = "sub_sc_altitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return f"Sub-observer altitude above {self.target} ellipsoid from {self.observer}"


# ---------------------------------------------------------------------------
# Sub-observer point — Rectangular (body-fixed XYZ)
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class SubObserverRectangular(TargetedProperty):
    """Sub-observer point expressed in body-fixed rectangular (x, y, z) coordinates (km).

    Components: ``sub_sc_x``, ``sub_sc_y``, ``sub_sc_z``
    """

    _name: ClassVar[str] = "sub_sc_rectangular"
    _unit: ClassVar[list] = [pint.Unit("km"), pint.Unit("km"), pint.Unit("km")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"Sub-observer ({self.observer}) rectangular coordinates on "
            f"{self.target} [x, y, z]"
        )

    @property
    def x(self) -> Property:
        return ComponentSelector(self, 0, "sub_sc_x")

    @property
    def y(self) -> Property:
        return ComponentSelector(self, 1, "sub_sc_y")

    @property
    def z(self) -> Property:
        return ComponentSelector(self, 2, "sub_sc_z")


@define(repr=False, order=False, eq=False)
class SubObserverX(TargetedProperty):
    """Body-fixed X coordinate of the sub-observer point (km)."""

    _name: ClassVar[str] = "sub_sc_x"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)



@define(repr=False, order=False, eq=False)
class SubObserverY(TargetedProperty):
    """Body-fixed Y coordinate of the sub-observer point (km)."""

    _name: ClassVar[str] = "sub_sc_y"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)



@define(repr=False, order=False, eq=False)
class SubObserverZ(TargetedProperty):
    """Body-fixed Z coordinate of the sub-observer point (km)."""

    _name: ClassVar[str] = "sub_sc_z"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    method: str = field(default=_SUBPNT_METHOD, kw_only=True)



# ---------------------------------------------------------------------------
# Boresight intersection — Latitudinal coordinates
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class BoresightIntersectionLatitudinal(TargetedProperty):
    """Boresight–surface intersection in latitudinal coordinates.

    Returns ``[NaN, NaN, NaN]`` when the boresight misses the body.

    Components: ``boresight_radius``, ``boresight_longitude``, ``boresight_latitude``
    """

    _name: ClassVar[str] = "boresight_latitudinal"
    _unit: ClassVar[list] = [pint.Unit("km"), pint.Unit("deg"), pint.Unit("deg")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    def __repr__(self) -> str:
        return (
            f"Boresight ({self.observer}) intersection latitudinal coordinates "
            f"on {self.target}"
        )

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "boresight_radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "boresight_longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 2, "boresight_latitude")


@define(repr=False, order=False, eq=False)
class BoresightLatitude(TargetedProperty):
    """Planetocentric latitude of the boresight–surface intersection (deg).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_latitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight latitude of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightLongitude(TargetedProperty):
    """Planetocentric longitude of the boresight–surface intersection (deg, positive east).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_longitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight longitude of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightRadius(TargetedProperty):
    """Distance from body centre to boresight–surface intersection (km).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_radius"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Boresight intersection radius of {self.observer} on {self.target}"


# ---------------------------------------------------------------------------
# Boresight intersection — Geodetic coordinates
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class BoresightIntersectionGeodetic(TargetedProperty):
    """Boresight–surface intersection in geodetic coordinates.

    Returns ``[NaN, NaN, NaN]`` when the boresight misses the body.

    Components: ``boresight_geo_longitude``, ``boresight_geo_latitude``, ``boresight_altitude``
    """

    _name: ClassVar[str] = "boresight_geodetic"
    _unit: ClassVar[list] = [pint.Unit("deg"), pint.Unit("deg"), pint.Unit("km")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    def __repr__(self) -> str:
        return (
            f"Boresight ({self.observer}) intersection geodetic coordinates "
            f"on {self.target}"
        )

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "boresight_geo_longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "boresight_geo_latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "boresight_altitude")


@define(repr=False, order=False, eq=False)
class BoresightGeoLatitude(TargetedProperty):
    """Geodetic latitude of the boresight–surface intersection (deg).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_geo_latitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight geodetic latitude of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightGeoLongitude(TargetedProperty):
    """Geodetic longitude of the boresight–surface intersection (deg).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_geo_longitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight geodetic longitude of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightAltitude(TargetedProperty):
    """Geodetic altitude of the boresight–surface intersection above reference ellipsoid (km).

    Returns ``NaN`` when the boresight misses the body.
    """

    _name: ClassVar[str] = "boresight_altitude"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Boresight intersection altitude of {self.observer} on {self.target}"


# ---------------------------------------------------------------------------
# Boresight intersection — Rectangular body-fixed XYZ
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class BoresightIntersectionRectangular(TargetedProperty):
    """Boresight–surface intersection in body-fixed rectangular coordinates (km).

    Returns ``[NaN, NaN, NaN]`` when the boresight misses the body.

    Components: ``boresight_x``, ``boresight_y``, ``boresight_z``
    """

    _name: ClassVar[str] = "boresight_rectangular"
    _unit: ClassVar[list] = [pint.Unit("km"), pint.Unit("km"), pint.Unit("km")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR

    def __repr__(self) -> str:
        return (
            f"Boresight ({self.observer}) intersection rectangular coordinates "
            f"on {self.target}"
        )

    @property
    def x(self) -> Property:
        return ComponentSelector(self, 0, "boresight_x")

    @property
    def y(self) -> Property:
        return ComponentSelector(self, 1, "boresight_y")

    @property
    def z(self) -> Property:
        return ComponentSelector(self, 2, "boresight_z")


@define(repr=False, order=False, eq=False)
class BoresightX(TargetedProperty):
    """Body-fixed X coordinate of the boresight–surface intersection (km)."""

    _name: ClassVar[str] = "boresight_x"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Boresight X of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightY(TargetedProperty):
    """Body-fixed Y coordinate of the boresight–surface intersection (km)."""

    _name: ClassVar[str] = "boresight_y"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Boresight Y of {self.observer} on {self.target}"


@define(repr=False, order=False, eq=False)
class BoresightZ(TargetedProperty):
    """Body-fixed Z coordinate of the boresight–surface intersection (km)."""

    _name: ClassVar[str] = "boresight_z"
    _unit: ClassVar[pint.Unit] = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Boresight Z of {self.observer} on {self.target}"


# ---------------------------------------------------------------------------
# Target RA/Dec (line-of-sight direction in J2000)
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class TargetRaDec(TargetedProperty):
    """Right ascension and declination of the target as seen from the observer (J2000, deg).

    Components: ``target_ra``, ``target_dec``
    """

    _name: ClassVar[str] = "target_radec"
    _unit: ClassVar[list] = [pint.Unit("deg"), pint.Unit("deg")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    def __repr__(self) -> str:
        return f"RA/Dec of {self.target} from {self.observer} (J2000)"

    @property
    def ra(self) -> Property:
        return ComponentSelector(self, 0, "target_ra")

    @property
    def dec(self) -> Property:
        return ComponentSelector(self, 1, "target_dec")


@define(repr=False, order=False, eq=False)
class TargetRA(TargetedProperty):
    """Right ascension of the target as seen from the observer (J2000, deg)."""

    _name: ClassVar[str] = "target_ra"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"RA of {self.target} from {self.observer} (J2000)"


@define(repr=False, order=False, eq=False)
class TargetDec(TargetedProperty):
    """Declination of the target as seen from the observer (J2000, deg)."""

    _name: ClassVar[str] = "target_dec"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Dec of {self.target} from {self.observer} (J2000)"


# ---------------------------------------------------------------------------
# Boresight RA/Dec (instrument pointing direction in J2000)
# ---------------------------------------------------------------------------

@define(repr=False, order=False, eq=False)
class BoresightRaDec(TargetedProperty):
    """Right ascension and declination of the instrument boresight in J2000 (deg).

    The ``target`` field is used only for context consistency and is not used
    in the computation.  Components: ``boresight_ra``, ``boresight_dec``
    """

    _name: ClassVar[str] = "boresight_radec"
    _unit: ClassVar[list] = [pint.Unit("deg"), pint.Unit("deg")]
    _type: ClassVar[PropertyTypes] = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    def __repr__(self) -> str:
        return f"Boresight RA/Dec of {self.observer} (J2000)"

    @property
    def ra(self) -> Property:
        return ComponentSelector(self, 0, "boresight_ra")

    @property
    def dec(self) -> Property:
        return ComponentSelector(self, 1, "boresight_dec")


@define(repr=False, order=False, eq=False)
class BoresightRA(TargetedProperty):
    """Right ascension of the instrument boresight in J2000 (deg)."""

    _name: ClassVar[str] = "boresight_ra"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight RA of {self.observer} (J2000)"


@define(repr=False, order=False, eq=False)
class BoresightDec(TargetedProperty):
    """Declination of the instrument boresight in J2000 (deg)."""

    _name: ClassVar[str] = "boresight_dec"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Boresight Dec of {self.observer} (J2000)"


# ---------------------------------------------------------------------------
# Illumination angle scalars (SubObserverIlluminationAngles components)
# ---------------------------------------------------------------------------
# These share a private helper that wraps the existing illum_angles call.
# The parent vector class (SubObserverIlluminationAngles) stays in
# observation_properties.py and remains unchanged.

from planetary_coverage.spice.toolbox import illum_angles as _illum_angles  # noqa: E402


def _sub_observer_illum(time, target, observer, abcorr: str) -> np.ndarray:
    """Return (incidence_deg, emission_deg, phase_deg) at the sub-observer point."""
    from ..properties.coordinates import SubObserverPoint

    pt = SubObserverPoint(observer, target)(time)
    i, e, p = _illum_angles(time, observer.spacecraft, target, pt, abcorr=abcorr, method="ELLIPSOID")
    return np.array([i, e, p])


@define(repr=False, order=False, eq=False)
class SubObserverIncidenceAngle(TargetedProperty):
    """Solar incidence angle at the sub-observer point (deg).

    0° = sun directly overhead, 90° = sun on horizon.
    """

    _name: ClassVar[str] = "sub_observer_incidence"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Sub-observer incidence angle on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverEmissionAngle(TargetedProperty):
    """Emission angle at the sub-observer point (deg).

    Angle between the surface normal and the direction to the observer.
    """

    _name: ClassVar[str] = "sub_observer_emission"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Sub-observer emission angle on {self.target} from {self.observer}"


@define(repr=False, order=False, eq=False)
class SubObserverPhaseAngleLocal(TargetedProperty):
    """Local phase angle at the sub-observer point (deg).

    Angle between the direction to the sun and the direction to the observer,
    as measured at the surface.  Different from the body-level phase angle
    (:class:`~spice_segmenter.properties.PhaseAngle`).
    """

    _name: ClassVar[str] = "sub_observer_phase_local"
    _unit: ClassVar[pint.Unit] = pint.Unit("deg")

    def __repr__(self) -> str:
        return f"Sub-observer local phase angle on {self.target} from {self.observer}"
