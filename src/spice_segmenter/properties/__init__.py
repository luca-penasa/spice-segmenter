"""Properties: concrete implementations for observation and motion constraints."""

from .occultation_types import Occultation, OccultationTypes
from .observation_properties import (
    AngularSize,
    ApproximatedAltitude,
    Distance,
    DistanceInTargetBodyRadii,
    MinMaxConditionTypes,
    PhaseAngle,
    SubObserverIlluminationAngles,
    SubObserverIsInDaylight,
    SubObserverPixelScale,
    SubObserverPointVelocity,
    TargetedProperty,
    TargetedPropertyMixin,
    TargetSizeOnSensor,
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
from .visibility_properties import BodyFOVVisibility, AngularSeparation
from .reflector_properties import JupiterShineIdealCondition
from .surface_properties import SurfaceIlluminationAngles

__all__ = [
    # Occultation
    "Occultation",
    "OccultationTypes",
    # Observation properties
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
    # Coordinates
    "Vector",
    "SubObserverPoint",
    "SubObserverPointMethods",
    "SphericalCoordinates",
    "CylindricalCoordinates",
    "LatitudinalCoordinates",
    "GeodeticCoordinates",
    "PlanetographicCoordinates",
    "RaDecCoordinates",
    # Visibility
    "BodyFOVVisibility",
    # Specialized
    "JupiterShineIdealCondition",
    "SurfaceIlluminationAngles",
]
