import pint
from attrs import define

from .trajectory_properties import Property


@define(repr=False, order=False, eq=False)
class LocalMaximum(Property):
    parent: Property
    max_distance: float = 0.0

    @property
    def name(self) -> str:
        return "local_maximum"

    @property
    def unit(self) -> pint.Unit:
        return self.parent.unit

    def __call__(self, time):
        return self.parent(time)

    def config(self, config: dict):
        config["operator"] = "local_maximum"
        config["max_distance"] = self.max_distance
