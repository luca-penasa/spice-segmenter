from typing import Tuple

import numpy as np
import pint
import spiceypy
from attr import define, field
from planetary_coverage import SpiceRef

from .decorators import vectorize
from .trajectory_properties import Property, PropertyTypes
from .types import times_types


@define(repr=False, order=False, eq=False)
class Vector(Property):
    origin: SpiceRef = field(converter=SpiceRef)
    target: SpiceRef = field(converter=SpiceRef)
    frame: str = field(default="J2000")
    abcorr: str = field(default="NONE")

    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "coordinate"

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("km"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.array:
        return spiceypy.spkpos(
            self.target.name, time, self.frame, self.abcorr, self.origin.name
        )[0]

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

    def config(self, config: dict) -> None:
        config.update(
            dict(
                origin=self.origin.name,
                target=self.target.name,
                frame=self.frame,
                abcorr=self.abcorr,
            )
        )
        config["vector_definition"] = "position"
        config["property"] = self.name
        config["coordinate_type"] = "rectangular"
        config["method"] = "ellipsoid"


@define(repr=False, order=False, eq=False)
class LatitudinalCoordinates(Property):
    vector: Vector

    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return f"latitudinal"

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
        return np.array(spiceypy.reclat(self.vector.__call__(time)))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 2, "latitude")

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class SphericalCoordinates(Property):
    vector: Vector

    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "spherical"

    @property
    def unit(self) -> Tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
        return np.array(spiceypy.recsph(self.vector.__call__(time)))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def colatitude(self) -> Property:
        return ComponentSelector(self, 1, "colatitude")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 2, "longitude")

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class CylindricalCoordinates(Property):
    vector: Vector

    def type(self):
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "cylindrical"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
        return np.array(spiceypy.reccyl(self.vector.__call__(time)))

    @property
    def radius(self) -> Property:
        return ComponentSelector(self, 0, "radius")

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 1, "longitude")

    @property
    def z(self) -> Property:
        return ComponentSelector(self, 2, "z")

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class GeodeticCoordinates(Property):
    vector: Vector

    def type(self):
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "geodetic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
        return np.array(spiceypy.recgeo(self.vector.__call__(time)))

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "altitude")

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class PlanetographicCoordinates(Property):
    vector: Vector

    def type(self):
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return "planetographic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
        return np.array(spiceypy.recpgr(self.vector.__call__(time)))

    @property
    def longitude(self) -> Property:
        return ComponentSelector(self, 0, "longitude")

    @property
    def latitude(self) -> Property:
        return ComponentSelector(self, 1, "latitude")

    @property
    def altitude(self) -> Property:
        return ComponentSelector(self, 2, "altitude")

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


@define(repr=False, order=False, eq=False)
class RaDecCoordinates(Property):
    vector: Vector

    def type(self):
        return PropertyTypes.VECTOR

    @property
    def name(self) -> str:
        return f"ra/dec"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: times_types) -> np.ndarray:
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

    def config(self, config: dict):
        self.vector.config(config)
        config["coordinate_type"] = self.name


from attrs.validators import instance_of


@define(repr=False, order=False, eq=False)
class ComponentSelector(Property):
    vector: Property = field(default=None)
    component: int = field(default=0, converter=int)
    _name: str = "component_selector"

    @vector.validator
    def _validate_vector(self, attribute, value):
        if not value.type() == PropertyTypes.VECTOR:
            raise ValueError(f"Vector must be of type {PropertyTypes.VECTOR}")

        return instance_of(Property)(self, attribute, value)

    def type(self):
        return PropertyTypes.SCALAR

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> pint.Unit:
        return self.vector.unit[self.component]

    @vectorize()
    def __call__(self, time: times_types) -> float:
        return self.vector.__call__(time)[self.component]

    def config(self, config: dict):
        self.vector.config(config)
        config["component"] = self.name
        config["property_unit"] = str(self.unit)
