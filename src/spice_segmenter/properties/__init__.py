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
from .geometry_properties import (
    BoresightAltitude,
    BoresightDec,
    # Boresight illumination angle scalars
    BoresightEmissionAngle,
    BoresightGeoLatitude,
    BoresightGeoLongitude,
    BoresightIncidenceAngle,
    # Boresight intersection geodetic
    BoresightIntersectionGeodetic,
    # Boresight intersection latitudinal
    BoresightIntersectionLatitudinal,
    # Boresight intersection rectangular
    BoresightIntersectionRectangular,
    BoresightLatitude,
    BoresightLongitude,
    BoresightPhaseAngleLocal,
    BoresightRA,
    # Boresight RA/Dec
    BoresightRaDec,
    BoresightRadius,
    # Boresight separation from a target
    BoresightSeparation,
    BoresightSlantRange,
    BoresightX,
    BoresightY,
    BoresightZ,
    SubObserverAltitude,
    SubObserverEmissionAngle,
    # Sub-observer geodetic
    SubObserverGeodetic,
    SubObserverGeoLatitude,
    SubObserverGeoLongitude,
    # Illumination angle scalars
    SubObserverIncidenceAngle,
    SubObserverLatitude,
    # Sub-observer latitudinal
    SubObserverLatitudinal,
    SubObserverLongitude,
    SubObserverPhaseAngleLocal,
    SubObserverRadius,
    # Sub-observer rectangular
    SubObserverRectangular,
    SubObserverX,
    SubObserverY,
    SubObserverZ,
    TargetDec,
    TargetRA,
    # Target RA/Dec
    TargetRaDec,
)
from .observation_properties import (
    AngularSize,
    ApproximatedAltitude,
    BoresightGroundtrackVelocity,
    Distance,
    DistanceInTargetBodyRadii,
    GroundtrackAngularRate,
    MinMaxConditionTypes,
    PhaseAngle,
    PushbroomRepetitionTime,
    SolarElongation,
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

__all__ = [
    # Occultation
    "Occultation",
    "OccultationTypes",
    # Observation properties
    "Distance",
    "PhaseAngle",
    "SolarElongation",
    "AngularSize",
    "TargetSizeOnSensor",
    "AngularSeparation",
    "SubObserverPointVelocity",
    "GroundtrackAngularRate",
    "PushbroomRepetitionTime",
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
    "BoresightSlantRange",
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
    # Geometry — boresight separation from a target
    "BoresightSeparation",
    # Geometry — illumination scalars
    "SubObserverIncidenceAngle",
    "SubObserverEmissionAngle",
    "SubObserverPhaseAngleLocal",
    "BoresightIncidenceAngle",
    "BoresightEmissionAngle",
    "BoresightPhaseAngleLocal",
    # Visibility
    "BodyFOVVisibility",
    # Specialized
    "JupiterShineIdealCondition",
    "SurfaceIlluminationAngles",
]
