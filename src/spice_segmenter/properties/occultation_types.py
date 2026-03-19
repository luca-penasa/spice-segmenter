from enum import Enum
from typing import ClassVar

import pint
import spiceypy
from attrs import define, field
from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceRef,
    SpiceSpacecraft,
)

from ..core.property import Property, PropertyTypes
from ..support.decorators import vectorize
from ..support.spice_utilities import et
from ..support.time_types import TIMES_TYPES


def _to_pint_unit(x):
    """Converter for unit fields."""
    from ..core.property import _to_pint_unit as _core_to_pint_unit
    return _core_to_pint_unit(x)


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
    _name: ClassVar[str] = "occultation"
    unit: pint.Unit = field(default=pint.Unit(""), kw_only=True, converter=_to_pint_unit)
    _type: ClassVar[PropertyTypes] = PropertyTypes.DISCRETE

    observer: SpiceSpacecraft | SpiceBody | SpiceInstrument = field(converter=SpiceRef)
    front: SpiceSpacecraft | SpiceBody | SpiceInstrument = field(converter=SpiceRef)
    back: SpiceSpacecraft | SpiceBody | SpiceInstrument = field(converter=SpiceRef)
    light_time_correction: str = field(default="NONE")

    def __repr__(self) -> str:
        return f"Occultation of {self.back} by {self.front}, as seen by {self.observer}"
