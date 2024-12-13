"""
Helper for conditional segmentation of a trajectory.

Warning
-------

This is just a stub, that might never see the light
"""

import importlib
import sys
import pandas as pd
from attrs import define, field
from loguru import logger as log

# from .constraint import Constraint

# from .constant import Constant

from .property_base import Property
from .constraint import ConstraintBase

from .coordinates import (
    CylindricalCoordinates,
    GeodeticCoordinates,
    LatitudinalCoordinates,
    PlanetographicCoordinates,
    RaDecCoordinates,
    SphericalCoordinates,
    SubObserverPoint,
    SubObserverPointMethods,
    Vector,
)
from .occultation import Occultation, OccultationTypes
from .ops import MinMaxConditionTypes, MinMaxConstraint
from .spice_window import SpiceWindow
from .trajectory_properties import (
    AngularSize,
    Distance,
    PhaseAngle,
)


from .constraint import Constraint

__version__ = importlib.metadata.version("spice_segmenter")

__all__ = [
    "MinMaxConstraint",
    "MinMaxConditionTypes",
    "SubObserverPointMethods",
    "SubObserverPoint",
    "LatitudinalCoordinates",
    "SphericalCoordinates",
    "Vector",
    "GeodeticCoordinates",
    "RaDecCoordinates",
    "CylindricalCoordinates",
    "PlanetographicCoordinates",
    "Distance",
    "PhaseAngle",
    "Occultation",
    "OccultationTypes",
    "SpiceWindow",
    "AngularSize",
    "constraint",
    "constant",
    "Constraint",
    "ConstraintBase",
    "Property"
]

log.disable("spice_segmenter")


def log_enable(
    level: str = "INFO",
    mod: str = "spice_segmenter",
    remove_handlers: bool = True,
) -> None:
    """Enable logging for a given module at specific level, by default it operates on the whole module."""
    if remove_handlers:
        log.remove()
    log.enable(mod)
    log.add(sys.stderr, level=level)


def log_enable_debug() -> None:
    """Enable debug logging for a given module, by default it operates on the whole module."""
    log_enable(level="DEBUG")


def log_disable(mod: str = "spice_segmenter") -> None:
    """Totally disable logging from this module, by default it operates on the whole module."""
    log.disable(mod)



def is_any_number(value):
    # Check if the value is a numeric type
    if isinstance(value, (int, float, complex)):
        return True

    # Check if the value is a string that can be converted to a number
    if isinstance(value, str):
        try:
            float(value)  # Attempt to convert to a float
            return True
        except ValueError:
            return False

    return False




def to_seconds(timedelta: str | pd.Timedelta | float | int):
    if is_any_number(timedelta):
        return float(timedelta)
    else:
        return pd.Timedelta(timedelta).total_seconds()

@define
class Config:
    """Configuration for the spice_segmenter module"""

    show_progressbar: bool = field(default=False)
    solver_step: float = field(default=5 * 60, converter=to_seconds)


config = Config()
