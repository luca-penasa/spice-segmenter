from typing import Any, Iterable, Optional

import pint
from attrs import define

from spice_segmenter.spice_window import SpiceWindow

from .trajectory_properties import Constraint, Property


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
        self.parent.config(config)
        config["operator"] = "local_maximum"
        config["max_distance"] = self.max_distance


@define(repr=False, order=False, eq=False)
class Inverted(Property):
    parent: Optional[
        Constraint
    ] = None  # need a validator to check is actually a constraint!

    @property
    def name(self) -> str:
        return f"{self.parent.name}"

    @property
    def unit(self) -> Any | Iterable:
        return self.parent.unit

    def solve(self, window: SpiceWindow, **kwargs) -> SpiceWindow:
        return self.parent.solve(window, **kwargs).complement(window)

    def __call__(self, time):
        return ~self.parent(time)
