"""SPICE-based trajectory event segmentation."""

import importlib
import sys

from loguru import logger as log

from .collections import (
    OccultationProperties,
    TargetProperties,
)

# Import order matters to avoid circular imports:
# 1. Core abstractions first
# 2. Then properties (which depend on core)
# 3. Then ops (which depend on core)
# 4. Then optimizers (which depend on ops and properties)
# 5. Then collections (which depend on properties)
from .core import (
    BooleanProperty,
    Constraint,
    ConstraintBase,
    ConstraintTypes,
    Property,
    PropertyTypes,
    SpiceWindow,
)
from .ops import (
    Constant,
    Inverted,
    MinMaxConstraint,
    UnitAdaptor,
)
from .optimizers import (
    ConstraintOptimizer,
    get_optimizer,
    optimize_constraint,
)
from .properties import (
    AngularSeparation,
    AngularSize,
    ApproximatedAltitude,
    BodyFOVVisibility,
    CylindricalCoordinates,
    Distance,
    DistanceInTargetBodyRadii,
    GeodeticCoordinates,
    JupiterShineIdealCondition,
    LatitudinalCoordinates,
    MinMaxConditionTypes,
    Occultation,
    OccultationTypes,
    PhaseAngle,
    PlanetographicCoordinates,
    RaDecCoordinates,
    SphericalCoordinates,
    SubObserverIlluminationAngles,
    SubObserverIsInDaylight,
    SubObserverPixelScale,
    SubObserverPoint,
    SubObserverPointMethods,
    SubObserverPointVelocity,
    SurfaceIlluminationAngles,
    TargetedProperty,
    TargetedPropertyMixin,
    TargetSizeOnSensor,
    Vector,
)
from .support.config import config
from .support.serialization import (
    create_property_converter,
    structure_constraint,
    unstructure_constraint,
)

# Get version
__version__ = importlib.metadata.version("spice_segmenter")

# Public API - organized by category
__all__ = [
    # Version
    "__version__",
    # Core abstractions
    "Property",
    "BooleanProperty",
    "PropertyTypes",
    "Constraint",
    "ConstraintBase",
    "ConstraintTypes",
    "SpiceWindow",
    # Properties - Observation
    "Distance",
    "PhaseAngle",
    "AngularSize",
    "TargetSizeOnSensor",
    "AngularSeparation",
    "SubObserverPointVelocity",
    "SubObserverPixelScale",
    "ApproximatedAltitude",
    "DistanceInTargetBodyRadii",
    "SubObserverIlluminationAngles",
    "SubObserverIsInDaylight",
    "MinMaxConditionTypes",
    "TargetedProperty",
    "TargetedPropertyMixin",
    # Properties - Coordinates
    "Vector",
    "SubObserverPoint",
    "SubObserverPointMethods",
    "SphericalCoordinates",
    "CylindricalCoordinates",
    "LatitudinalCoordinates",
    "GeodeticCoordinates",
    "PlanetographicCoordinates",
    "RaDecCoordinates",
    # Properties - Occultation
    "Occultation",
    "OccultationTypes",
    # Properties - Specialized
    "BodyFOVVisibility",
    "JupiterShineIdealCondition",
    "SurfaceIlluminationAngles",
    # Constraint Operations
    "Constant",
    "UnitAdaptor",
    "MinMaxConstraint",
    "Inverted",
    # Optimization
    "ConstraintOptimizer",
    "optimize_constraint",
    "get_optimizer",
    # Collections
    "TargetProperties",
    "OccultationProperties",
    # Serialization
    "create_property_converter",
    "structure_constraint",
    "unstructure_constraint",
    # Logging
    "log_enable",
    "log_disable",
    "log_enable_debug",
    # Configuration
    "config",
]

# Disable logging by default
log.disable("spice_segmenter")


def log_enable(
    level: str = "INFO",
    mod: str = "spice_segmenter",
    remove_handlers: bool = True,
) -> None:
    """
    Enable logging for the spice_segmenter module.
    
    Parameters
    ----------
    level : str, optional
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR'), by default 'INFO'
    mod : str, optional
        Module name to enable logging for, by default 'spice_segmenter'
    remove_handlers : bool, optional
        Whether to remove existing handlers before adding stderr, by default True
        
    Example
    -------
    >>> log_enable('DEBUG')
    >>> log_enable_debug()  # Shorthand
    """
    if remove_handlers:
        log.remove()
    log.enable(mod)
    log.add(sys.stderr, level=level)


def log_enable_debug() -> None:
    """Enable debug-level logging for spice_segmenter."""
    log_enable(level="DEBUG")


def log_disable(mod: str = "spice_segmenter") -> None:
    """
    Disable logging for the spice_segmenter module.
    
    Parameters
    ----------
    mod : str, optional
        Module name to disable logging for, by default 'spice_segmenter'
    """
    log.disable(mod)

