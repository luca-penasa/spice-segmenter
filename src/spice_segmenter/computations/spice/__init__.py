"""SPICE computation layer — registers all built-in compute functions.

``register_all(engine)`` is called exactly once, lazily, from
:func:`~spice_segmenter.engines.evaluator.get_evaluator`.  Every
class object, so a missing import raises ``ImportError`` immediately rather
than silently failing at evaluation time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spice_segmenter.properties.observation_properties import RelativeSpeed

if TYPE_CHECKING:
    from ...engines.spice_engine import SpiceEngine


def register_all(engine: SpiceEngine) -> None:
    """Populate *engine* with all built-in SPICE compute functions.

    Imports are intentionally local to this function so that circular-import
    issues at module-load time are completely avoided (the whole
    ``computations/`` tree is only imported on first use).
    """

    # ------------------------------------------------------------------
    # Observation properties
    # ------------------------------------------------------------------
    from ...properties.observation_properties import (
        AngularSize,
        ApproximatedAltitude,
        Distance,
        DistanceInTargetBodyRadii,
        PhaseAngle,
        SubObserverPixelScale,
        TargetSizeOnSensor,
    )
    from .observation import (
        angular_size_scalar,
        angular_size_vector,
        approx_altitude_scalar,
        approx_altitude_vector,
        distance_in_target_radii_scalar,
        distance_in_target_radii_vector,
        distance_scalar,
        distance_vector,
        phase_angle_scalar,
        phase_angle_vector,
        sub_observer_pixel_scale_scalar,
        sub_observer_pixel_scale_vector,
        target_size_on_sensor_scalar,
        target_size_on_sensor_vector,
        relative_speed_scalar
    )

    engine.register(Distance, scalar_fn=distance_scalar, vector_fn=distance_vector, compute_unit="km")
    engine.register(PhaseAngle, scalar_fn=phase_angle_scalar, vector_fn=phase_angle_vector, compute_unit="rad")
    engine.register(AngularSize, scalar_fn=angular_size_scalar, vector_fn=angular_size_vector, compute_unit="rad")
    engine.register(RelativeSpeed, scalar_fn=relative_speed_scalar, compute_unit="km/s")
    engine.register(
        ApproximatedAltitude,
        scalar_fn=approx_altitude_scalar,
        vector_fn=approx_altitude_vector,
        compute_unit="km",
    )
    engine.register(
        TargetSizeOnSensor,
        scalar_fn=target_size_on_sensor_scalar,
        vector_fn=target_size_on_sensor_vector,
        compute_unit="px",
    )
    engine.register(
        DistanceInTargetBodyRadii,
        scalar_fn=distance_in_target_radii_scalar,
        vector_fn=distance_in_target_radii_vector,
        compute_unit="",
    )
    engine.register(
        SubObserverPixelScale,
        scalar_fn=sub_observer_pixel_scale_scalar,
        vector_fn=sub_observer_pixel_scale_vector,
        compute_unit="km/px",
    )

    # ------------------------------------------------------------------
    # Occultation
    # ------------------------------------------------------------------
    from ...properties.occultation_types import Occultation
    from .occultation import occultation_scalar, occultation_vector

    engine.register(Occultation, scalar_fn=occultation_scalar, vector_fn=occultation_vector, compute_unit="")

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------
    from ...properties.visibility_properties import BodyFOVVisibility
    from .visibility import fov_visibility_scalar, fov_visibility_vector

    engine.register(
        BodyFOVVisibility,
        scalar_fn=fov_visibility_scalar,
        vector_fn=fov_visibility_vector,
    )

    # ------------------------------------------------------------------
    # Geometry properties (sub-observer point + boresight intersection)
    # ------------------------------------------------------------------
    from ...properties.geometry_properties import (
        BoresightAltitude,
        BoresightDec,
        BoresightGeoLatitude,
        BoresightGeoLongitude,
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
        SubObserverAltitude,
        SubObserverGeodetic,
        SubObserverGeoLatitude,
        SubObserverGeoLongitude,
        SubObserverLatitude,
        SubObserverLatitudinal,
        SubObserverLongitude,
        SubObserverRadius,
        SubObserverRectangular,
        SubObserverX,
        SubObserverY,
        SubObserverZ,
        TargetDec,
        TargetRA,
        TargetRaDec,
    )
    from .geometry import (
        boresight_altitude_scalar,
        boresight_dec_scalar,
        boresight_geo_latitude_scalar,
        boresight_geo_longitude_scalar,
        boresight_geodetic_scalar,
        boresight_geodetic_vector,
        boresight_latitude_scalar,
        boresight_latitudinal_scalar,
        boresight_latitudinal_vector,
        boresight_longitude_scalar,
        boresight_ra_scalar,
        boresight_radec_scalar,
        boresight_radec_vector,
        boresight_radius_scalar,
        boresight_rectangular_scalar,
        boresight_rectangular_vector,
        boresight_x_scalar,
        boresight_y_scalar,
        boresight_z_scalar,
        sub_sc_altitude_scalar,
        sub_sc_altitude_vector,
        sub_sc_geo_latitude_scalar,
        sub_sc_geo_latitude_vector,
        sub_sc_geo_longitude_scalar,
        sub_sc_geo_longitude_vector,
        sub_sc_geodetic_scalar,
        sub_sc_geodetic_vector,
        sub_sc_latitude_scalar,
        sub_sc_latitude_vector,
        sub_sc_latitudinal_scalar,
        sub_sc_latitudinal_vector,
        sub_sc_longitude_scalar,
        sub_sc_longitude_vector,
        sub_sc_radius_scalar,
        sub_sc_radius_vector,
        sub_sc_rectangular_scalar,
        sub_sc_rectangular_vector,
        sub_sc_x_scalar,
        sub_sc_x_vector,
        sub_sc_y_scalar,
        sub_sc_y_vector,
        sub_sc_z_scalar,
        sub_sc_z_vector,
        target_dec_scalar,
        target_ra_scalar,
        target_radec_scalar,
        target_radec_vector,
    )

    # Sub-observer latitudinal
    engine.register(
        SubObserverLatitudinal,
        scalar_fn=sub_sc_latitudinal_scalar,
        vector_fn=sub_sc_latitudinal_vector,
    )
    engine.register(SubObserverRadius, scalar_fn=sub_sc_radius_scalar, vector_fn=sub_sc_radius_vector, compute_unit="km")
    engine.register(SubObserverLongitude, scalar_fn=sub_sc_longitude_scalar, vector_fn=sub_sc_longitude_vector, compute_unit="deg")
    engine.register(SubObserverLatitude, scalar_fn=sub_sc_latitude_scalar, vector_fn=sub_sc_latitude_vector, compute_unit="deg")

    # Sub-observer geodetic
    engine.register(
        SubObserverGeodetic,
        scalar_fn=sub_sc_geodetic_scalar,
        vector_fn=sub_sc_geodetic_vector,
    )
    engine.register(SubObserverGeoLongitude, scalar_fn=sub_sc_geo_longitude_scalar, vector_fn=sub_sc_geo_longitude_vector, compute_unit="deg")
    engine.register(SubObserverGeoLatitude, scalar_fn=sub_sc_geo_latitude_scalar, vector_fn=sub_sc_geo_latitude_vector, compute_unit="deg")
    engine.register(SubObserverAltitude, scalar_fn=sub_sc_altitude_scalar, vector_fn=sub_sc_altitude_vector, compute_unit="km")

    # Sub-observer rectangular
    engine.register(
        SubObserverRectangular,
        scalar_fn=sub_sc_rectangular_scalar,
        vector_fn=sub_sc_rectangular_vector,
    )
    engine.register(SubObserverX, scalar_fn=sub_sc_x_scalar, vector_fn=sub_sc_x_vector, compute_unit="km")
    engine.register(SubObserverY, scalar_fn=sub_sc_y_scalar, vector_fn=sub_sc_y_vector, compute_unit="km")
    engine.register(SubObserverZ, scalar_fn=sub_sc_z_scalar, vector_fn=sub_sc_z_vector, compute_unit="km")

    # Boresight latitudinal
    engine.register(
        BoresightIntersectionLatitudinal,
        scalar_fn=boresight_latitudinal_scalar,
        vector_fn=boresight_latitudinal_vector,
    )
    engine.register(BoresightLatitude, scalar_fn=boresight_latitude_scalar, compute_unit="deg")
    engine.register(BoresightLongitude, scalar_fn=boresight_longitude_scalar, compute_unit="deg")
    engine.register(BoresightRadius, scalar_fn=boresight_radius_scalar, compute_unit="km")

    # Boresight geodetic
    engine.register(
        BoresightIntersectionGeodetic,
        scalar_fn=boresight_geodetic_scalar,
        vector_fn=boresight_geodetic_vector,
    )
    engine.register(BoresightGeoLatitude, scalar_fn=boresight_geo_latitude_scalar, compute_unit="deg")
    engine.register(BoresightGeoLongitude, scalar_fn=boresight_geo_longitude_scalar, compute_unit="deg")
    engine.register(BoresightAltitude, scalar_fn=boresight_altitude_scalar, compute_unit="km")

    # Boresight rectangular
    engine.register(
        BoresightIntersectionRectangular,
        scalar_fn=boresight_rectangular_scalar,
        vector_fn=boresight_rectangular_vector,
    )
    engine.register(BoresightX, scalar_fn=boresight_x_scalar, compute_unit="km")
    engine.register(BoresightY, scalar_fn=boresight_y_scalar, compute_unit="km")
    engine.register(BoresightZ, scalar_fn=boresight_z_scalar, compute_unit="km")

    # Target RA/Dec
    engine.register(TargetRaDec, scalar_fn=target_radec_scalar, vector_fn=target_radec_vector, compute_unit=("deg", "deg"))
    engine.register(TargetRA, scalar_fn=target_ra_scalar, compute_unit="deg")
    engine.register(TargetDec, scalar_fn=target_dec_scalar, compute_unit="deg")

    # Boresight RA/Dec
    engine.register(BoresightRaDec, scalar_fn=boresight_radec_scalar, vector_fn=boresight_radec_vector, compute_unit=("deg", "deg"))
    engine.register(BoresightRA, scalar_fn=boresight_ra_scalar, compute_unit="deg")
    engine.register(BoresightDec, scalar_fn=boresight_dec_scalar, compute_unit="deg")

    # ------------------------------------------------------------------
    # SubObserver illumination angle scalars  (geometry module)
    # ------------------------------------------------------------------
    from ...properties.geometry_properties import (
        SubObserverEmissionAngle,
        SubObserverIncidenceAngle,
        SubObserverPhaseAngleLocal,
    )
    from .geometry import (
        sub_observer_emission_angle_scalar,
        sub_observer_incidence_angle_scalar,
        sub_observer_phase_angle_local_scalar,
    )

    engine.register(SubObserverIncidenceAngle, scalar_fn=sub_observer_incidence_angle_scalar, compute_unit="deg")
    engine.register(SubObserverEmissionAngle, scalar_fn=sub_observer_emission_angle_scalar, compute_unit="deg")
    engine.register(SubObserverPhaseAngleLocal, scalar_fn=sub_observer_phase_angle_local_scalar, compute_unit="deg")

    # ------------------------------------------------------------------
    # Observation: SubObserverIlluminationAngles, SubObserverIsInDaylight,
    #              AngularSeparation, SubObserverPointVelocity
    # ------------------------------------------------------------------
    from ...properties.observation_properties import (
        SubObserverIlluminationAngles,
        SubObserverIsInDaylight,
        SubObserverPointVelocity,
    )
    from ...properties.visibility_properties import AngularSeparation
    from .observation import (
        angular_separation_scalar,
        sub_observer_illumination_angles_scalar,
        sub_observer_is_in_daylight_scalar,
        sub_observer_point_velocity_scalar,
    )

    engine.register(
        SubObserverIlluminationAngles,
        scalar_fn=sub_observer_illumination_angles_scalar,
    )
    engine.register(SubObserverIsInDaylight, scalar_fn=sub_observer_is_in_daylight_scalar, compute_unit="")
    engine.register(AngularSeparation, scalar_fn=angular_separation_scalar, compute_unit="rad")
    engine.register(SubObserverPointVelocity, scalar_fn=sub_observer_point_velocity_scalar, compute_unit="km/s")

    # ------------------------------------------------------------------
    # Visibility: AngularSeparation is actually in observation_properties
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Coordinates: Vector, SubObserverPoint, Boresight, BoresightIntersection,
    #              BoresightIntersects, and all coordinate-transform classes
    # ------------------------------------------------------------------
    from ...properties.coordinates import (
        Boresight,
        BoresightIntersection,
        BoresightIntersects,
        CylindricalCoordinates,
        GeodeticCoordinates,
        LatitudinalCoordinates,
        PlanetographicCoordinates,
        RaDecCoordinates,
        SphericalCoordinates,
        SubObserverPoint,
        Vector,
    )
    from .coordinates import (
        boresight_intersection_scalar,
        boresight_intersects_scalar,
        boresight_scalar,
        cylindrical_scalar,
        geodetic_scalar,
        latitudinal_scalar,
        planetographic_scalar,
        radec_scalar,
        spherical_scalar,
        sub_observer_point_scalar,
        sub_observer_point_vector,
        vector_scalar,
        vector_vector,
    )

    engine.register(Vector, scalar_fn=vector_scalar, vector_fn=vector_vector, compute_unit=("km", "km", "km"))
    engine.register(
        SubObserverPoint,
        scalar_fn=sub_observer_point_scalar,
        vector_fn=sub_observer_point_vector,
    )
    engine.register(Boresight, scalar_fn=boresight_scalar, compute_unit=("km", "km", "km"))
    engine.register(BoresightIntersection, scalar_fn=boresight_intersection_scalar, compute_unit=("km", "km", "km"))
    engine.register(BoresightIntersects, scalar_fn=boresight_intersects_scalar, compute_unit="")
    engine.register(LatitudinalCoordinates, scalar_fn=latitudinal_scalar, compute_unit=("km", "deg", "deg"))
    engine.register(SphericalCoordinates, scalar_fn=spherical_scalar, compute_unit=("km", "deg", "deg"))
    engine.register(CylindricalCoordinates, scalar_fn=cylindrical_scalar, compute_unit=("km", "deg", "km"))
    engine.register(GeodeticCoordinates, scalar_fn=geodetic_scalar, compute_unit=("deg", "deg", "km"))
    engine.register(PlanetographicCoordinates, scalar_fn=planetographic_scalar, compute_unit=("deg", "deg", "km"))
    engine.register(RaDecCoordinates, scalar_fn=radec_scalar, compute_unit=("km", "deg", "deg"))

    # ------------------------------------------------------------------
    # Reflector (Jupiter shine)
    # ------------------------------------------------------------------
    from ...properties.reflector_properties import (
        JupiterRise,
        JupiterRiseRatio,
        ShineProperties,
    )
    from .reflector import (
        jupiter_rise_ratio_scalar,
        jupiter_rise_scalar,
        shine_properties_scalar,
    )

    engine.register(ShineProperties, scalar_fn=shine_properties_scalar, compute_unit=("deg", "deg", "deg", "deg"))
    engine.register(JupiterRise, scalar_fn=jupiter_rise_scalar, compute_unit="")
    engine.register(JupiterRiseRatio, scalar_fn=jupiter_rise_ratio_scalar, compute_unit="")

    # ------------------------------------------------------------------
    # Ring properties
    # ------------------------------------------------------------------
    from ...properties.ring_properties import (
        RingAnsaePhaseGreaterThan,
        RingAnsaePhaseLowerThan,
        RingAnsaePhaseWithinRange,
    )
    from .ring import (
        ring_greater_than_scalar,
        ring_lower_than_scalar,
        ring_within_range_scalar,
    )

    engine.register(RingAnsaePhaseLowerThan, scalar_fn=ring_lower_than_scalar, compute_unit="")
    engine.register(RingAnsaePhaseGreaterThan, scalar_fn=ring_greater_than_scalar, compute_unit="")
    engine.register(RingAnsaePhaseWithinRange, scalar_fn=ring_within_range_scalar, compute_unit="")
