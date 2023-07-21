from typing import Iterable, Union

from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceObserver,
    SpiceSpacecraft,
)

TIME_TYPES = Union[str, float, int]
TIMES_TYPES = Union[TIME_TYPES, Iterable[TIME_TYPES]]
obj_type = Union[SpiceBody, SpiceInstrument, SpiceSpacecraft, SpiceObserver]
