"""Properties: concrete implementations for observation and motion constraints."""

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
from .observation_properties import (
    AngularSize,
    ApproximatedAltitude,
    BoresightGroundtrackVelocity,
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
from .occultation_types import Occultation, OccultationTypes
from .reflector_properties import JupiterShineIdealCondition
from .surface_properties import SurfaceIlluminationAngles
from .visibility_properties import AngularSeparation, BodyFOVVisibility
from .geometry_properties import (
    # Sub-observer latitudinal
    SubObserverLatitudinal,
    SubObserverLatitude,
    SubObserverLongitude,
    SubObserverRadius,
    # Sub-observer geodetic
    SubObserverGeodetic,
    SubObserverGeoLatitude,
    SubObserverGeoLongitude,
    SubObserverAltitude,
    # Sub-observer rectangular
    SubObserverRectangular,
    SubObserverX,
    SubObserverY,
    SubObserverZ,
    # Boresight intersection latitudinal
    BoresightIntersectionLatitudinal,
    BoresightLatitude,
    BoresightLongitude,
    BoresightRadius,
    # Boresight intersection geodetic
    BoresightIntersectionGeodetic,
    BoresightGeoLatitude,
    BoresightGeoLongitude,
    BoresightAltitude,
    # Boresight intersection rectangular
    BoresightIntersectionRectangular,
    BoresightX,
    BoresightY,
    BoresightZ,
    # Target RA/Dec
    TargetRaDec,
    TargetRA,
    TargetDec,
    # Boresight RA/Dec
    BoresightRaDec,
    BoresightRA,
    BoresightDec,
    # Illumination angle scalars
    SubObserverIncidenceAngle,
    SubObserverEmissionAngle,
    SubObserverPhaseAngleLocal,
)

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
    "BoresightGroundtrackVelocity",
    "SubObserverPixelScale",
    "ApproximatedAltitude",
    "DistanceInTargetBodyRadii",
    "SubObserverIlluminationAngles",
    "SubObserverIsInDaylight",
    "MinMaxConditionTypes",
    "TargetedProperty",
    "TargetedPropertyMixin",
    # Coords
    "Vector",
    "SubObserverPoint",
    "SubObserverPointMethods",
    "SphericalCoordinates",
    "CylindricalCoordinates",
    "LatitudinalCoordinates",
    "GeodeticCoordinates",
    "PlanetographicCoordinates",
    "RaDecCoordinates",
    # Geometry — sub-observer latitudinal
    "SubObserverLatitudinal",
    "SubObserverLatitude",
    "SubObserverLongitude",
    "SubObserverRadius",
    # Geometry — sub-observer geodetic
    "SubObserverGeodetic",
    "SubObserverGeoLatitude",
    "SubObserverGeoLongitude",
    "SubObserverAltitude",
    # Geometry — sub-observer rectangular
    "SubObserverRectangular",
    "SubObserverX",
    "SubObserverY",
    "SubObserverZ",
    # Geometry — boresight latitudinal
    "BoresightIntersectionLatitudinal",
    "BoresightLatitude",
    "BoresightLongitude",
    "BoresightRadius",
    # Geometry — boresight geodetic
    "BoresightIntersectionGeodetic",
    "BoresightGeoLatitude",
    "BoresightGeoLongitude",
    "BoresightAltitude",
    # Geometry — boresight rectangular
    "BoresightIntersectionRectangular",
    "BoresightX",
    "BoresightY",
    "BoresightZ",
    # Geometry — RA/Dec
    "TargetRaDec",
    "TargetRA",
    "TargetDec",
    "BoresightRaDec",
    "BoresightRA",
    "BoresightDec",
    # Geometry — illumination scalars
    "SubObserverIncidenceAngle",
    "SubObserverEmissionAngle",
    "SubObserverPhaseAngleLocal",
    # Visibility
    "BodyFOVVisibility",
    # Specialized
    "JupiterShineIdealCondition",
    "SurfaceIlluminationAngles",
]
