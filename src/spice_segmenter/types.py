from typing import Union

import numpy as np
from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceObserver,
    SpiceSpacecraft,
)

times_types = Union[str, float, list, tuple, np.ndarray]
obj_type = Union[SpiceBody, SpiceInstrument, SpiceSpacecraft, SpiceObserver]
