
from typing import ClassVar

import pint
from pydantic import Field

from spice_segmenter.support.model import segmenter_dataclass
from spice_segmenter.support.types import (
    PropertyNode,
    SpiceBodyField,
    SpiceFrameField,
    SpiceInstrumentField,
    SpiceRefField,
)

from ..core.property import BooleanProperty, Property, PropertyTypes
from ..properties.component_selector import ComponentSelector
from ..support.context import get_current_observer, get_current_target


@segmenter_dataclass
class VectorBase(Property):
    frame: SpiceFrameField = Field(default="J2000", kw_only=True)
    abcorr: str = Field(default="NONE", kw_only=True)
    # Scalar __call__ returns a 3-element array; this enables np.vectorize fallback.
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

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
    def unit(self) -> tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("km"), pint.Unit("km")

    def config(self, config: dict) -> None:
        config.update(
            {
                "frame": self.frame.name,
                "abcorr": self.abcorr,
            },
        )


@segmenter_dataclass
class Vector(VectorBase):
    origin: SpiceRefField
    target: SpiceRefField

    def __repr__(self) -> str:
        return f"Vector from {self.origin} to {self.target} in frame {self.frame}"

    @property
    def name(self) -> str:
        return "coordinate"

    def config(self, config: dict) -> None:
        super().config(config)
        config.update(
            {
                "origin": self.origin.name,
                "target": self.target.name,
            },
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


@segmenter_dataclass
class SubObserverPoint(Vector):
    method: str = SubObserverPointMethods.INTERCEPT

    def __repr__(self) -> str:
        return f"Sub observer ({self.origin}) point on {self.target} in frame {self.frame}, computed using {self.method} method"

    def config(self, config: dict) -> None:
        super().config(config)
        config["method"] = str(self.method)


@segmenter_dataclass
class Boresight(VectorBase):
    instrument: SpiceInstrumentField

    @property
    def name(self) -> str:
        return "boresight"

    def __repr__(self) -> str:
        return f"Boresight of {self.instrument} in frame {self.frame}"

    def config(self, config: dict) -> None:
        super().config(config)
        config["instrument"] = self.instrument.name


@segmenter_dataclass
class BoresightIntersection(Boresight):
    target: SpiceBodyField
    boresight: Boresight | None = Field(default=None, init=False)

    def __post_init__(self):
        self.boresight = Boresight(self.instrument, frame=self.frame)

    @property
    def name(self) -> str:
        return "boresight_intersection"

    def __repr__(self) -> str:
        return f"Boresight intersection of {self.instrument} on {self.target} (computed in frame {self.frame})"

    def config(self, config: dict) -> None:
        super().config(config)
        config["target"] = self.target.name


@segmenter_dataclass
class BoresightIntersects(BooleanProperty):
    _name = "boresight_intersects_body"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN

    observer: SpiceInstrumentField = Field(default_factory=get_current_observer)
    target: SpiceBodyField = Field(default_factory=get_current_target)

    intersection: BoresightIntersection | None = Field(init=False, default=None)

    def __post_init__(self):
        self.intersection = BoresightIntersection(
            target=self.target, instrument=self.observer,
        )

    def __repr__(self) -> str:
        return f"Boresight of {self.observer} intersects {self.target}"


@segmenter_dataclass
class LatitudinalCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "latitudinal"

    @property
    def unit(self) -> tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

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


@segmenter_dataclass
class SphericalCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "spherical"

    @property
    def unit(self) -> tuple[pint.Unit, pint.Unit, pint.Unit]:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

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


@segmenter_dataclass
class CylindricalCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "cylindrical"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("km")

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


@segmenter_dataclass
class GeodeticCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "geodetic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

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


@segmenter_dataclass
class PlanetographicCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "planetographic"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad"), pint.Unit("rad"), pint.Unit("km")

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


@segmenter_dataclass
class RaDecCoordinates(Property):
    vector: PropertyNode

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.VECTOR

    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    @property
    def name(self) -> str:
        return "ra/dec"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km"), pint.Unit("rad"), pint.Unit("rad")

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["coordinate_type"] = self.name




