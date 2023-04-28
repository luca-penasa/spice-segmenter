from enum import Enum

import pint
import spiceypy
from attrs import define, field
from planetary_coverage.spice import SpiceRef
from planetary_coverage.spice.times import et

from spice_segmenter.decorators import vectorize
from spice_segmenter.trajectory_properties import Property, PropertyTypes
from spice_segmenter.types import times_types


class OccultationTypes(Enum):
    NONE = 0
    FULL = 1
    PARTIAL = 2
    ANNULAR = 3
    ANY = 5


@define(repr=False, order=False, eq=False)
class Occulatation(Property):
    observer: SpiceRef = field(converter=SpiceRef)
    front: SpiceRef = field(converter=SpiceRef)
    back: SpiceRef = field(converter=SpiceRef)
    light_time_correction: str = field(default="NONE")

    @property
    def name(self) -> str:
        return f"occultation"

    def type(self) -> PropertyTypes:
        return PropertyTypes.DISCRETE

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("")

    def _remap_to_enum(self, value: int) -> OccultationTypes:
        if value == -3:
            return OccultationTypes.FULL
        elif value == -2:
            return OccultationTypes.ANNULAR
        elif value == -1:
            return OccultationTypes.PARTIAL
        else:
            return OccultationTypes.NONE

    @vectorize
    def __call__(self, times: times_types) -> OccultationTypes:
        v = spiceypy.occult(
            self.back.name,
            "ELLIPSOID",
            self.back.frame,
            self.front.name,
            "ELLIPSOID",
            self.front.frame,
            self.light_time_correction,
            self.observer.name,
            et(times),
        )

        return self._remap_to_enum(v)
