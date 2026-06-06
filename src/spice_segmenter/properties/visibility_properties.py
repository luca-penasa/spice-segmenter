
import pint
from attrs import define, field
from planetary_coverage.spice import SpiceRef

from ..core.property import PropertyTypes
from ..properties.observation_properties import TargetedProperty


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
