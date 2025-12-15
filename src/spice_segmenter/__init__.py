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

# onlyu after here to avoid circular imports
from .collections import OccultationProperties, TargetProperties
from .constant import Constant
from .constraint import Constraint, ConstraintBase
from .constraint_optimizer import (
    ConstraintOptimizer,
    get_optimizer,
    optimize_constraint,
)
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

# from .constraint import Constraint
# from .constant import Constant
from .property_base import Property
from .serialization import (
    create_property_converter,
    structure_constraint,
    unstructure_constraint,
)
from .spice_window import SpiceWindow
from .trajectory_properties import (
    AngularSize,
    Distance,
    PhaseAngle,
    TargetSizeOnSensor,
    angular_size_to_distance,
    pixel_count_to_distance,
)

__version__ = importlib.metadata.version("spice_segmenter")

__all__ = [
    "AngularSize",
    "Constant",
    "Constraint",
    "ConstraintBase",
    "ConstraintOptimizer",
    "CylindricalCoordinates",
    "Distance",
    "GeodeticCoordinates",
    "LatitudinalCoordinates",
    "MinMaxConditionTypes",
    "MinMaxConstraint",
    "Occultation",
    "OccultationProperties",
    "OccultationTypes",
    "PhaseAngle",
    "PlanetographicCoordinates",
    "Property",
    "RaDecCoordinates",
    "SphericalCoordinates",
    "SpiceWindow",
    "SubObserverPoint",
    "SubObserverPointMethods",
    "TargetProperties",
    "TargetSizeOnSensor",
    "Vector",
    "angular_size_to_distance",
    "constant",
    "constraint",
    "create_property_converter",
    "get_optimizer",
    "optimize_constraint",
    "pixel_count_to_distance",
    "structure_constraint",
    "unstructure_constraint",
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


def to_seconds(timedelta: str | pd.Timedelta | float):
    if is_any_number(timedelta):
        return float(timedelta)
    return pd.Timedelta(timedelta).total_seconds()


@define
class Config:
    """Configuration for the spice_segmenter module"""

    show_progressbar: bool = field(default=False)
    solver_step: float = field(default=5 * 60, converter=to_seconds)


config = Config()
