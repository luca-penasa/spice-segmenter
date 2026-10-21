from enum import Enum
from typing import ClassVar

import pint
from pydantic import Field

from spice_segmenter.support.model import segmenter_dataclass
from spice_segmenter.support.types import SpiceRefField, UnitField

from ..core.property import Property, PropertyTypes


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


@segmenter_dataclass
class Occultation(Property):
    _name: ClassVar[str] = "occultation"
    unit: UnitField = Field(default=pint.Unit(""), kw_only=True)
    _type: ClassVar[PropertyTypes] = PropertyTypes.DISCRETE

    observer: SpiceRefField
    front: SpiceRefField
    back: SpiceRefField
    light_time_correction: str = "NONE"

    def __repr__(self) -> str:
        return f"Occultation of {self.back} by {self.front}, as seen by {self.observer}"
