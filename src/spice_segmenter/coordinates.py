from typing import Any, Tuple

import numpy as np
import pint
import spiceypy
from attr import define, field
from numpy.typing import ArrayLike
from planetary_coverage import SpiceRef, et
from planetary_coverage.spice import SpiceBody, SpiceFrame, SpiceInstrument
from spiceypy import NotFoundError

from .decorators import vectorize
from .trajectory_properties import Property, PropertyTypes
from .types import TIMES_TYPES


@define(repr=False, order=False, eq=False)
class VectorBase(Property):
    frame: SpiceFrame = field(default="J2000", kw_only=True, converter=SpiceFrame)
    abcorr: str = field(default="NONE", kw_only=True)

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def x(self) -> Property:
        return ComponentSelector(self, 0, "x")

    @property
    def y(self) -> Property:
        return ComponentSelector(self, 1, "y")

    @property
    def z(self) -> Property:
        return ComponentSelector(self, 2, "z")

    @property
    def as_latitudinal(self) -> "LatitudinalCoordinates":
        return LatitudinalCoordinates(self)

    @property
    def as_radec(self) -> "RaDecCoordinates":
        return RaDecCoordinates(self)

    @property
    def as_spherical(self) -> "SphericalCoordinates":
        return SphericalCoordinates(self)

    @property
    def as_cylindrical(self) -> "CylindricalCoordinates":
        return CylindricalCoordinates(self)

    @property
    def as_geodetic(self) -> "GeodeticCoordinates":
        return GeodeticCoordinates(self)

    @property
    def as_planetographic(self) -> "PlanetographicCoordinates":
        return PlanetographicCoordinates(self)

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("km"), pint.Unit("km")

    def config(self, config: dict) -> None:
        config.update(
            {
                "frame": self.frame.name,
                "abcorr": self.abcorr,
            }
        )


@define(repr=False, order=False, eq=False)
class Vector(VectorBase):
    origin: SpiceRef = field(converter=SpiceRef)
    target: SpiceRef = field(converter=SpiceRef)

    def __repr__(self) -> str:
        return f"Vector from {self.origin} to {self.target} in frame {self.frame}"

    @property
    def name(self) -> str:
        return "coordinate"

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> ArrayLike:
        return spiceypy.spkpos(
            self.target.name, et(time), self.frame, self.abcorr, self.origin.name
        )[0]

    def config(self, config: dict) -> None:
        super().config(config)
        config.update(
            {
                "origin": self.origin.name,
                "target": self.target.name,
            }
        )
        config["vector_definition"] = "position"
        config["property"] = self.name
        config["coordinate_type"] = "rectangular"
        config["method"] = "ellipsoid"


class SubObserverPointMethods:
    """
    Sub-observer point computation methods, for now we support only these two.
    """

    NEAREST = "NEAR POINT/ELLIPSOID"
    INTERCEPT = "INTERCEPT/ELLIPSOID"


@define(repr=False, order=False, eq=False)
class SubObserverPoint(Vector):
    method = field(default=SubObserverPointMethods.INTERCEPT)

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        return spiceypy.subpnt(
            str(self.method),
            self.target.name,
            et(time),
            self.target.frame.name,
            "None",
            self.origin.name,
        )[0]

    def __repr__(self) -> str:
        return f"Sub observer ({self.origin}) point on {self.target} in frame {self.frame}, computed using {self.method} method"

    def config(self, config: dict) -> None:
        super().config(config)
        config["method"] = str(self.method)


@define(repr=False, order=False, eq=False)
class Boresight(VectorBase):
    instrument = field(converter=SpiceInstrument)

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        T = spiceypy.pxform(self.instrument.frame, self.frame, et(time))
        return T @ np.array([0, 0, 1])  # we should get this value from spices, please

    @property
    def name(self) -> str:
        return "boresight"

    def __repr__(self) -> str:
        return f"Boresight of {self.instrument} in frame {self.frame}"

    def config(self, config: dict) -> None:
        super().config(config)
        config["instrument"] = self.instrument.name


@define(repr=False, order=False, eq=False)
class BoresightIntersection(Boresight):
    target = field(converter=SpiceBody)
    boresight = field(default=None, init=False)

    def __attrs_post_init__(self):
        self.boresight = Boresight(self.instrument, frame=self.frame)

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        bsight = self.boresight(time)
        try:
            return spiceypy.sincpt(
                "ELLIPSOID",
                self.target.name,
                et(time),
                self.target.frame.name,
                self.abcorr,
                self.instrument.name,
                self.frame,
                bsight,
            )[0]
        except NotFoundError:
            return np.array([np.nan, np.nan, np.nan])

    @property
    def name(self) -> str:
        return "boresight_intersection"

    def __repr__(self) -> str:
        return f"Boresight intersection of {self.instrument} on {self.target} in frame {self.frame}"

    def config(self, config: dict) -> None:
        super().config(config)
        config["target"] = self.target.name


@define(repr=False, order=False, eq=False)
class LatitudinalCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "latitudinal"

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        value = self.vector.__call__(time)
        if np.isnan(value).any():
            return np.array([np.nan, np.nan, np.nan])
        return np.array(spiceypy.reclat(value))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 2, "latitude")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class SphericalCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "spherical"

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        value = self.vector.__call__(time)
        if np.isnan(value).any():
            return np.array([np.nan, np.nan, np.nan])
        return np.array(spiceypy.recsph(value))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def colatitude(self) -> Property:
        return ComponentSelector(self, 1, "colatitude")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 2, "longitude")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class CylindricalCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "cylindrical"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        value = self.vector.__call__(time)
        if np.isnan(value).any():
            return np.array([np.nan, np.nan, np.nan])
        return np.array(spiceypy.reccyl(value))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "longitude")

    @property
    def z(self) -> Property:
        return ComponentSelector(self, 2, "z")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class GeodeticCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "geodetic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        value = self.vector.__call__(time)
        if np.isnan(value).any():
            return np.array([np.nan, np.nan, np.nan])
        return np.array(spiceypy.recgeo(value))

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "altitude")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class PlanetographicCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "planetographic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        value = self.vector.__call__(time)
        if np.isnan(value).any():
            return np.array([np.nan, np.nan, np.nan])
        else:
            return np.array(
                spiceypy.recpgr(
                    self.vector.target.name,
                    value,
                    self.vector.target.re,
                    self.vector.target.f,
                )
            )

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "altitude")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class RaDecCoordinates(Property):
    vector: Vector

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "ra/dec"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        return np.array(spiceypy.recrad(self.vector.__call__(time)))

    @property
    def range(self) -> Property:
        return ComponentSelector(self, 0, "range")

    @property
    def right_ascension(self) -> Property:
        return ComponentSelector(self, 1, "right_ascension")

    @property
    def declination(self) -> Property:
        return ComponentSelector(self, 2, "declination")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name


from attrs.validators import instance_of


@define(repr=False, order=False, eq=False)
class ComponentSelector(Property):
    vector: Property = field(default=None)
    component: int = field(default=0, converter=int)
    _name: str = "component_selector"

    @vector.validator
    def _validate_vector(self, attribute, value) -> Any:  # type: ignore
        if not value.type == PropertyTypes.VECTOR:
            raise ValueError(f"Vector must be of type {PropertyTypes.VECTOR}")

        return instance_of(Property)(self, attribute, value)

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.SCALAR

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> pint.Unit:
        return self.vector.unit[self.component]

    @vectorize()
    def __call__(self, time: TIMES_TYPES) -> float:
        return self.vector.__call__(time)[self.component]

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["component"] = self.name
        config["property_unit"] = str(self.unit)
