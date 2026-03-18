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


@define(repr=False, order=False, eq=False)
class BodyFOVVisibility(TargetedProperty):
    """Visibility of a body from the FOV of an instrument"""
    
    _name = "fov_visibility"
    _type = PropertyTypes.BOOLEAN
    _unit = pint.Unit("")

    def __repr__(self) -> str:
        return f"Visibility of {self.target} from {self.observer} FOV"
    

    


@define(repr=False, order=False, eq=False)
class AngularSeparation(TargetedProperty):
    """Angular separation between two bodies"""
    
    _name = "angular_separation"
    _type = PropertyTypes.SCALAR
    _unit = pint.Unit("rad")

    other = field(converter=SpiceRef, kw_only=True)

    def __repr__(self) -> str:
        return f"Angular separation between {self.target} and {self.observer}"

    def config(self, config: dict) -> None:
        super().config(config)
        config.update({"other": self.other.name})
        config.update({"other_frame": self.other.frame.name})
