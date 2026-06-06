"""SPICE compute functions for observation properties.

Each function has the signature::

    fn(prop: ConcretePropertyClass, time_et: float) -> value   # scalar
    fn(prop: ConcretePropertyClass, times_et: ndarray) -> ndarray  # vector

The *prop* argument carries all configuration fields (observer, target,
light_time_correction, etc.) so the functions are pure — they do not rely on
any global state beyond SPICE kernel availability.

Derived properties (AngularSize, ApproximatedAltitude, TargetSizeOnSensor,
DistanceInTargetBodyRadii, SubObserverPixelScale) call the Distance helpers
directly rather than round-tripping through the engine for efficiency.
"""

from __future__ import annotations
from os import times

import numpy as np
import spiceypy

from ...properties.observation_properties import (
    AngularSize,
    ApproximatedAltitude,
    Distance,
    DistanceInTargetBodyRadii,
    PhaseAngle,
    SubObserverPixelScale,
    TargetSizeOnSensor,
    RelativeSpeed,
)

from planetary_coverage import et

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def distance_scalar(prop: Distance, time_et: float) -> float:
    return spiceypy.vnorm(
        spiceypy.spkpos(
            prop.target.name,
            time_et,
            prop.observer.frame.name,
            prop.light_time_correction,
            prop.observer.name,
        )[0],
    )


def distance_vector(prop: Distance, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice

    positions, _ = cyice.spkpos_v(
        prop.target.name,
        times_et,
        prop.observer.frame.name,
        prop.light_time_correction,
        prop.observer.name,
    )
    return np.linalg.norm(positions, axis=1)


# ---------------------------------------------------------------------------
# PhaseAngle
# ---------------------------------------------------------------------------


def phase_angle_scalar(prop: PhaseAngle, time_et: float) -> float:
    return spiceypy.phaseq(
        time_et,
        prop.target.name,
        prop.third_body.name,
        prop.observer.name,
        prop.light_time_correction,
    )


def phase_angle_vector(prop: PhaseAngle, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice

    return cyice.phaseq_v(
        times_et,
        prop.target.name,
        prop.third_body.name,
        prop.observer.name,
        prop.light_time_correction,
    )


# ---------------------------------------------------------------------------
# AngularSize  (derived: calls distance helpers directly)
# ---------------------------------------------------------------------------


def angular_size_scalar(prop: AngularSize, time_et: float) -> float:
    d = distance_scalar(prop, time_et)
    return float(2 * np.arctan(prop.target.radius / d))


def angular_size_vector(prop: AngularSize, times_et: np.ndarray) -> np.ndarray:
    distances = distance_vector(prop, times_et)
    return 2 * np.arctan(prop.target.radius / distances)


# ---------------------------------------------------------------------------
# ApproximatedAltitude  (derived)
# ---------------------------------------------------------------------------


def approx_altitude_scalar(prop: ApproximatedAltitude, time_et: float) -> float:
    return distance_scalar(prop, time_et) - prop.target.radius


def approx_altitude_vector(
    prop: ApproximatedAltitude,
    times_et: np.ndarray,
) -> np.ndarray:
    return distance_vector(prop, times_et) - prop.target.radius


# ---------------------------------------------------------------------------
# TargetSizeOnSensor  (derived)
# ---------------------------------------------------------------------------


def target_size_on_sensor_scalar(prop: TargetSizeOnSensor, time_et: float) -> float:
    return angular_size_scalar(prop, time_et) / np.mean(prop.observer.ifov)


def target_size_on_sensor_vector(
    prop: TargetSizeOnSensor,
    times_et: np.ndarray,
) -> np.ndarray:
    return angular_size_vector(prop, times_et) / np.mean(prop.observer.ifov)


# ---------------------------------------------------------------------------
# DistanceInTargetBodyRadii  (derived)
# ---------------------------------------------------------------------------


def distance_in_target_radii_scalar(
    prop: DistanceInTargetBodyRadii,
    time_et: float,
) -> float:
    return distance_scalar(prop, time_et) / prop.target.radius


def distance_in_target_radii_vector(
    prop: DistanceInTargetBodyRadii,
    times_et: np.ndarray,
) -> np.ndarray:
    return distance_vector(prop, times_et) / prop.target.radius


# ---------------------------------------------------------------------------
# SubObserverPixelScale  (derived — calls pixel_scale helper)
# ---------------------------------------------------------------------------


def sub_observer_pixel_scale_scalar(
    prop: SubObserverPixelScale,
    time_et: float,
) -> float:
    from planetary_coverage.spice.toolbox import pixel_scale

    d = distance_scalar(prop, time_et)
    return pixel_scale(inst=prop.observer, target=prop.target, emi=0, dist=d)


def sub_observer_pixel_scale_vector(
    prop: SubObserverPixelScale,
    times_et: np.ndarray,
) -> np.ndarray:
    from planetary_coverage.spice.toolbox import pixel_scale

    distances = distance_vector(prop, times_et)
    return np.vectorize(
        lambda d: pixel_scale(inst=prop.observer, target=prop.target, emi=0, dist=d),
    )(distances)


# ---------------------------------------------------------------------------
# SubObserverIlluminationAngles  (vector output: [incidence, emission, phase])
# ---------------------------------------------------------------------------


def sub_observer_illumination_angles_scalar(
    prop,
    time_et: float,
) -> np.ndarray:
    from planetary_coverage.spice.toolbox import illum_angles

    from ...engines.evaluator import get_evaluator
    from ...properties.coordinates import SubObserverPoint

    sub_pt = SubObserverPoint(prop.observer, prop.target)
    pt = get_evaluator().evaluate_scalar(sub_pt, time_et)
    result = illum_angles(
        time_et,
        prop.observer.spacecraft,
        prop.target,
        pt,
        abcorr=prop.light_time_correction,
        method="ELLIPSOID",
    )
    return np.array(result)


# ---------------------------------------------------------------------------
# SubObserverIsInDaylight  (boolean — incidence < 90°)
# ---------------------------------------------------------------------------


def sub_observer_is_in_daylight_scalar(prop, time_et: float) -> bool:
    from ...engines.evaluator import get_evaluator
    from ...properties.observation_properties import SubObserverIlluminationAngles

    illum = SubObserverIlluminationAngles(prop.observer, prop.target)
    angles = get_evaluator().evaluate_scalar(illum, time_et)
    return bool(angles[0] < 90.0)


# ---------------------------------------------------------------------------
# AngularSeparation  (vsep of two position vectors)
# ---------------------------------------------------------------------------


def angular_separation_scalar(prop, time_et: float) -> float:
    # Match original: Vector() uses J2000 frame and NONE abcorr by default
    pos1 = spiceypy.spkpos(
        prop.target.name,
        time_et,
        "J2000",
        "NONE",
        prop.observer.name,
    )[0]
    pos2 = spiceypy.spkpos(
        prop.other.name,
        time_et,
        "J2000",
        "NONE",
        prop.observer.name,
    )[0]
    return float(spiceypy.vsep(pos1, pos2))


# ---------------------------------------------------------------------------
# SubObserverPointVelocity  (groundtrack velocity at sub-observer point)
# ---------------------------------------------------------------------------


def sub_observer_point_velocity_scalar(prop, time_et: float) -> float:
    from planetary_coverage.spice.toolbox import groundtrack_velocity, sc_state

    state = sc_state(
        time_et,
        prop.observer.spacecraft,
        prop.target,
        prop.light_time_correction,
    )
    return groundtrack_velocity(prop.target, state)


def relative_speed_scalar(prop: RelativeSpeed, time_et: float) -> float:
    """Return |v_rel| (km/s) of target w.r.t. observer for each input time."""

    state, _ = spiceypy.spkezr(
        prop.target, et(time_et), "J2000", prop.light_time_correction, prop.observer
    )
    return np.linalg.norm(state[3:6], axis=0)
