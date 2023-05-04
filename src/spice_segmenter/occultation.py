from enum import Enum

import pint
import spiceypy
from attrs import define, field
from planetary_coverage.spice import SpiceRef

from spice_segmenter.decorators import vectorize
from spice_segmenter.trajectory_properties import Property, PropertyTypes
from spice_segmenter.types import times_types

from .utils import et


class OccultationTypes(Enum):
    NONE = 0
    FULL = 1
    PARTIAL = 2
    ANNULAR = 3
    ANY = 5

    def __repr__(self) -> str:
        return "%s" % (self._name_)

    def __str__(self) -> str:
        return "%s" % (self._name_)

    def __eq__(self, other: object) -> bool:
        """Custom comparison for occultation taking into account that ANY should match any other type not None

        Probably this implementation could be improved in a smarter way. we do have tests for it if needed
        """
        if self.__class__ is other.__class__:
            if other.value == OccultationTypes.ANY.value:
                return True if self.value != OccultationTypes.NONE.value else False

            if self.value == OccultationTypes.ANY.value:
                return True if other.value != OccultationTypes.NONE.value else False

        return super().__eq__(other)


@define(repr=False, order=False, eq=False)
class Occultation(Property):
    observer: SpiceRef = field(converter=SpiceRef)
    front: SpiceRef = field(converter=SpiceRef)
    back: SpiceRef = field(converter=SpiceRef)
    light_time_correction: str = field(default="NONE")

    def __repr__(self) -> str:
        return f"Occultation of {self.back} by {self.front}, as seen by {self.observer}"

    @property
    def name(self) -> str:
        return "occultation"

    @property
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
