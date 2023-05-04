from typing import Iterable, Union

from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceObserver,
    SpiceSpacecraft,
)

times_types = Union[str, float, int] | Iterable[Union[str, float, int]]
obj_type = Union[SpiceBody, SpiceInstrument, SpiceSpacecraft, SpiceObserver]
