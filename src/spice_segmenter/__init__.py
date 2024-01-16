"""
Helper for conditional segmentation of a trajectory.

Warning
-------

This is just a stub, that might never see the light
"""

from attr import define, field

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
    Constant,
    Constraint,
    Distance,
    PhaseAngle,
)

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
    "Constraint",
    "Constant",
]

import sys

from loguru import logger

# entirely disables logging for the spice_segmenter module
logger.disable("spice_segmenter")
logger.remove()

logger.add(sys.stderr, level="WARNING")


@define
class Config:
    """Configuration for the spice_segmenter module"""

    show_progressbar: bool = field(default=True)
    solver_step: float = field(default=60 * 60)


config = Config()
