"""SPICE-based trajectory event segmentation."""

import importlib
import sys

from loguru import logger as log

from .collections import (
    OccultationProperties,
    PropertySnapshot,
    TargetProperties,
    compute_all,
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
    TimeSegment,
    TimeSegmentsCollection,
    all_properties,
    get_property,
    property_registry,
)
from .io.dsl import (
    constraint_to_context,
    constraint_to_expression,
)
from .io.dsl import (
    parse as parse_constraint,
)
from .io.yaml_io import (
    dump as dump_constraint,
)
from .io.yaml_io import (
    dump_properties,
    dumps_properties,
    load_properties,
    loads_properties,
)
from .io.yaml_io import (
    dumps as dumps_constraint,
)
from .io.yaml_io import (
    load as load_constraint,
)
from .io.yaml_io import (
    loads as loads_constraint,
)
from .ops import (
    Constant,
    Inverted,
    MinMaxConstraint,
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
    BoresightAltitude,
    BoresightDec,
    BoresightGeoLatitude,
    BoresightGeoLongitude,
    BoresightGroundtrackVelocity,
    BoresightIntersectionGeodetic,
    BoresightIntersectionLatitudinal,
    BoresightIntersectionRectangular,
    BoresightLatitude,
    BoresightLongitude,
    BoresightRA,
    BoresightRaDec,
    BoresightRadius,
    BoresightX,
    BoresightY,
    BoresightZ,
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
    SubObserverAltitude,
    SubObserverEmissionAngle,
    SubObserverGeodetic,
    SubObserverGeoLatitude,
    SubObserverGeoLongitude,
    SubObserverIlluminationAngles,
    SubObserverIncidenceAngle,
    SubObserverIsInDaylight,
    SubObserverLatitude,
    SubObserverLatitudinal,
    SubObserverLongitude,
    SubObserverPhaseAngleLocal,
    SubObserverPixelScale,
    SubObserverPoint,
    SubObserverPointMethods,
    SubObserverPointVelocity,
    SubObserverRadius,
    SubObserverRectangular,
    SubObserverX,
    SubObserverY,
    SubObserverZ,
    SurfaceIlluminationAngles,
    TargetDec,
    TargetedProperty,
    TargetedPropertyMixin,
    TargetRA,
    TargetRaDec,
    TargetSizeOnSensor,
    Vector,
)
from .support.config import Config, config, get_active_config
from .support.context import (
    SpiceContext,
    get_active_context,
    get_context,
    get_current_light_time_correction,
    get_current_observer,
    get_current_target,
    spice_context,
)
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
    "BoresightGroundtrackVelocity",
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
    # DSL
    "parse_constraint",
    "constraint_to_expression",
    "constraint_to_context",
    # YAML I/O
    "load_constraint",
    "loads_constraint",
    "dump_constraint",
    "dumps_constraint",
    "load_properties",
    "loads_properties",
    "dump_properties",
    "dumps_properties",
    # Logging
    "log_enable",
    "log_disable",
    "log_enable_debug",
    # Registry
    "property_registry",
    "get_property",
    "all_properties",
    # Configuration
    "Config",
    "config",
    "get_active_config",
    # Context management
    "SpiceContext",
    "spice_context",
    "get_active_context",
    "get_context",
    "get_current_observer",
    "get_current_target",
    "get_current_light_time_correction",
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


def get_property_registry() -> dict:
    """Deprecated: use ``property_registry`` or ``all_properties()`` instead."""
    return all_properties()
