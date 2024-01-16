from enum import Enum

import pint
import spiceypy
from attr import field
from planetary_coverage.spice import SpiceRef

from spice_segmenter.coordinates import Vector
from spice_segmenter.decorators import declare, vectorize
from spice_segmenter.trajectory_properties import PropertyTypes, TargetedProperty
from spice_segmenter.types import TIMES_TYPES
from spice_segmenter.utils import et


@declare(name="fov_visibility", property_type=PropertyTypes.BOOLEAN)
class BodyFOVVisibility(TargetedProperty):
    """Visibility of a body from the FOV of an instrument"""

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


@declare(
    name="angular_separation", property_type=PropertyTypes.SCALAR, unit=pint.Unit("rad")
)
class AngularSeparation(TargetedProperty):
    """Angular separation between two bodies"""

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
