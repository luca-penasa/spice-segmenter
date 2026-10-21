
import pint
from pydantic import Field

from spice_segmenter.support.model import segmenter_dataclass
from spice_segmenter.support.types import SpiceRefField

from ..core.property import PropertyTypes
from ..properties.observation_properties import TargetedProperty


@segmenter_dataclass
class BodyFOVVisibility(TargetedProperty):
    """Visibility of a body from the FOV of an instrument"""

    _name = "fov_visibility"
    _type = PropertyTypes.BOOLEAN
    _unit = pint.Unit("")

    def __repr__(self) -> str:
        return f"Visibility of {self.target} from {self.observer} FOV"





@segmenter_dataclass
class AngularSeparation(TargetedProperty):
    """Angular separation between two bodies"""

    _name = "angular_separation"
    _type = PropertyTypes.SCALAR
    _unit = pint.Unit("rad")

    other: SpiceRefField = Field(kw_only=True)

    def __repr__(self) -> str:
        return f"Angular separation between {self.target} and {self.observer}"

    def config(self, config: dict) -> None:
        super().config(config)
        config.update({"other": self.other.name})
        config.update({"other_frame": self.other.frame.name})
