from enum import Enum

import pint
import spiceypy
from attrs import define, field
from planetary_coverage.spice import SpiceRef

from ..core.property import PropertyTypes
from ..properties.coordinates import Vector
from ..properties.observation_properties import TargetedProperty
from ..support.decorators import vectorize
from ..support.spice_utilities import et
from ..support.time_types import TIMES_TYPES


class BodyFOVVisibility(TargetedProperty):
    """Visibility of a body from the FOV of an instrument"""
    
    _name = "fov_visibility"
    _type = PropertyTypes.BOOLEAN

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum:
        time = et(time)
        return spiceypy.fovtrg(
            self.observer.name,
            self.target.name,
            "ELLIPSOID",
            self.target.frame,
            self.light_time_correction,
            self.observer.name,
            time,
        )

    def __repr__(self) -> str:
        return f"Visibility of {self.target} from {self.observer} FOV"


@define(repr=False, order=False, eq=False)
class AngularSeparation(TargetedProperty):
    """Angular separation between two bodies"""
    
    _name = "angular_separation"
    _type = PropertyTypes.SCALAR
    _unit = pint.Unit("rad")

    other = field(converter=SpiceRef)

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum:
        time = et(time)
        v1 = Vector(self.observer, self.target)

        v2 = Vector(self.observer, self.other)

        return spiceypy.vsep(v1(time), v2(time))

    def __repr__(self) -> str:
        return f"Angular separation between {self.target} and {self.observer}"

    def config(self, config: dict) -> None:
        super().config(config)
        config.update({"other": self.other.name})
        config.update({"other_frame": self.other.frame.name})
