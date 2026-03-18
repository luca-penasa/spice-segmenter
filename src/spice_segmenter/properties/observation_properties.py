from __future__ import annotations

from enum import Enum
from typing import ClassVar

import numpy as np
import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define, field
from loguru import logger as log
from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceSpacecraft,
)
from planetary_coverage.spice.toolbox import (
    groundtrack_velocity,
    illum_angles,
    sc_state,
)

from spice_segmenter.core.property import Property, PropertyTypes
from spice_segmenter.properties.component_selector import ComponentSelector
from spice_segmenter.support.context import (
    get_current_light_time_correction,
    get_current_observer,
    get_current_target,
)
from spice_segmenter.support.decorators import vectorize
from spice_segmenter.support.spice_utilities import as_spice_ref, et
from spice_segmenter.support.time_types import TIMES_TYPES

PROPERTIES_REGISTRY = []


class MinMaxConditionTypes(Enum):
    LOCAL_MINIMUM = "local_minimum"
    LOCAL_MAXIMUM = "local_maximum"
    GLOBAL_MINIMUM = "global_minimum"
    GLOBAL_MAXIMUM = "global_maximum"


@define(repr=False, order=False, eq=False)
class TargetedPropertyMixin:
    observer: SpiceInstrument | SpiceSpacecraft = field(
        factory=get_current_observer,
        converter=as_spice_ref,
    )
    target: SpiceBody = field(factory=get_current_target, converter=as_spice_ref)
    light_time_correction: str = field(
        factory=get_current_light_time_correction,
        kw_only=True,
    )

    def config(self, config: dict) -> None:
        log.debug(
            "targeted property config here with instance of {}",
            self.__class__.__name__,
        )
        super().config(config)
        config.update(
            {
                "target": self.target.name,
                "target_frame": self.target.frame.name,
                "observer": self.observer.name,
                "abcorr": self.light_time_correction,
            },
        )


@define(repr=False, order=False, eq=False)
class TargetedProperty(TargetedPropertyMixin, Property):
    pass


@define(repr=False, order=False, eq=False)
class PhaseAngle(TargetedProperty):
    _name = "phase_angle"
    _unit = pint.Unit("rad")

    third_body: SpiceBody = field(
        factory=lambda: as_spice_ref("SUN"),
        converter=as_spice_ref,
    )

    def __repr__(self) -> str:
        return f"Phase Angle of {self.target} with respect to {self.third_body} as seen from {self.observer}"

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["third_body"] = self.third_body.name
        config["property"] = self.name


@define(repr=False, order=False, eq=False)
class Distance(TargetedProperty):
    _name = "distance"
    _unit = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Distance of {self.target} from {self.observer}"

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define(repr=False, order=False, eq=False)
class SubObserverPointVelocity(TargetedProperty):
    _name = "sub_sc_velocity"
    _unit = pint.Unit("km/s")
    # method = field(default=SubObserverPointMethods.INTERCEPT)

    def __repr__(self) -> str:
        return f"Velocity of sub observer ({self.observer}) point on {self.target} surface."


@define
class BoresightGroundtrackVelocity(TargetedProperty):
    """Groundtrack velocity of the boresight intersection on the target surface.

    Computes how fast the instrument boresight footprint moves across
    the target body surface. Unlike :class:`SubObserverPointVelocity`
    (which tracks the sub-observer/nadir point), this property follows
    the actual boresight pointing direction and accounts for any
    spacecraft slew or off-nadir pointing.

    The velocity is obtained by numerically differentiating the
    boresight-surface intersection point (via SPICE ``sincpt``).

    Three computation methods are available (selectable via the
    ``method`` parameter):

    ``"pc"``
        Uses :func:`~planetary_coverage.spice.toolbox.groundtrack_velocity`
        which accounts for the full geodetic shape (oblate spheroid).
    ``"sphere"``
        Projects the velocity onto the local tangent plane of a sphere
        with the target's mean radius.
    ``"euclidean"``
        Returns the raw Euclidean norm of the finite-difference
        velocity vector (no surface projection).

    Returns ``NaN`` when the boresight does not intersect the target.
    """

    _name: ClassVar[str] = "boresight_groundtrack_velocity"
    _unit: ClassVar[pint.Unit] = pint.Unit("km/s")

    dt: float = field(
        default=1.0,
        kw_only=True,
    )  # time step (seconds) for central finite difference

    method: str = field(default="pc", kw_only=True)

    METHODS: ClassVar[tuple[str, ...]] = ("pc", "sphere", "euclidean")

    def __attrs_post_init__(self) -> None:
        if self.method not in self.METHODS:
            msg = f"Unknown method {self.method!r}. Choose from {self.METHODS}"
            raise ValueError(msg)

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        t = et(time)
        dt = self.dt

        p_minus = self._boresight_surface_point(t - dt / 2)
        p_plus = self._boresight_surface_point(t + dt / 2)

        if np.any(np.isnan(p_minus)) or np.any(np.isnan(p_plus)):
            return np.nan

        velocity = (p_plus - p_minus) / dt

        if self.method == "euclidean":
            return float(np.linalg.norm(velocity))

        if self.method == "sphere":
            position = (p_minus + p_plus) / 2
            # radial unit vector at the surface point
            r_hat = position / np.linalg.norm(position)
            # tangential velocity = reject the radial component
            v_tan = velocity - np.dot(velocity, r_hat) * r_hat
            return float(np.linalg.norm(v_tan))

        # method == "pc"
        position = (p_minus + p_plus) / 2
        state = np.concatenate([position, velocity])
        return groundtrack_velocity(self.target, state)

    def _boresight_surface_point(self, t_et: float) -> np.ndarray:
        """Compute boresight-surface intersection in the target body-fixed frame."""
        try:
            return spiceypy.sincpt(
                "ELLIPSOID",
                self.target.name,
                t_et,
                self.target.frame.name,
                self.light_time_correction,
                self.observer.name,
                str(self.observer.frame),
                np.array([0.0, 0.0, 1.0]),
            )[0]
        except (
            spiceypy.exceptions.NotFoundError,
            spiceypy.utils.exceptions.NotFoundError,
        ):
            return np.array([np.nan, np.nan, np.nan])

    def __repr__(self) -> str:
        return (
            f"Groundtrack velocity of {self.observer} boresight "
            f"intersection on {self.target} surface "
            f"(method={self.method!r})."
        )


@define
class AngularSize(TargetedProperty):
    _name = "angular_size"
    _unit = pint.Unit("rad")

    def __repr__(self) -> str:
        return f"Angular size of {self.target}, seen from {self.observer}"

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define
class SubObserverPixelScale(TargetedProperty):
    _name = "sub_observer_pixel_scale"
    _unit = pint.Unit("km/px")

    def __repr__(self) -> str:
        return f"Resultion of {self.target}, at the sub-{self.observer} point."

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define
class ApproximatedAltitude(TargetedProperty):
    _name = "approx_altitude"
    _unit = pint.Unit("km")

    def __repr__(self) -> str:
        return f"Approximated (distance-radius) altitude of {self.observer} over {self.target} surface (from sub-sc point)"

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define
class TargetSizeOnSensor(TargetedProperty):
    _name = "target_size_on_sensor"
    _unit = pint.Unit("px")

    def __repr__(self) -> str:
        return f"Diameter in pixels of {self.target}, on the {self.observer} sensor."

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define
class DistanceInTargetBodyRadii(TargetedProperty):
    _name = "distance_in_target_radii"
    _unit = pint.Unit("dimensionless")

    def __repr__(self) -> str:
        return f"Distance of {self.target}, from {self.observer} sensor, in {self.target} radii."

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@define
class SubObserverIlluminationAngles(TargetedProperty):
    _name = "sub_observer_illumination_angles"
    _unit = [pint.Unit("deg"), pint.Unit("deg"), pint.Unit("deg")]
    _type = PropertyTypes.VECTOR
    _vector_output_shape: ClassVar[str | None] = "()->(n)"

    def __repr__(self) -> str:
        return f"Sub {self.observer} point illumination angles on {self.target}"

    @property
    def incidence(self) -> Property:
        return ComponentSelector(self, 0, "incidence")

    @property
    def emission(self) -> Property:
        return ComponentSelector(self, 1, "emission")

    @property
    def phase(self) -> Property:
        return ComponentSelector(self, 2, "phase")


@define
class SubObserverIsInDaylight(TargetedProperty):
    _name = "sub_obs_point_in_daylight"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN

    def __repr__(self) -> str:
        return f"Sub {self.observer} point on {self.target} is in daylight"

    pass  # engine-registered; computation via evaluator


# Utility functions for constraint optimization


def pixel_count_to_distance(
    observer: SpiceInstrument | SpiceSpacecraft,
    target: SpiceBody,
    n_pixels: float,
) -> float:
    """Convert pixel count on sensor to distance threshold.

    This is used for constraint optimization: instead of computing expensive
    TargetSizeOnSensor properties, we can use a simple Distance constraint.

    Formula:
        angular_size = 2 * arctan(target_radius / distance)
        n_pixels = angular_size / pixel_scale
        distance = target_radius / tan(n_pixels * pixel_scale / 2)

    Args:
        observer: SpiceInstrument or SpiceSpacecraft
        target: SpiceBody target
        n_pixels: Number of pixels the target should span

    Returns:
        Distance in km where target spans n_pixels on sensor

    Examples:
        >>> dist_km = pixel_count_to_distance(
        ...     "JUICE_JANUS",
        ...     "METIS",
        ...     5,
        ... )
        >>> # Now Distance < dist_km replaces TargetSizeOnSensor > 5px
    """
    observer = as_spice_ref(observer)
    target = as_spice_ref(target)

    ifov = np.mean(observer.ifov)  # Average pixel scale in radians
    target_radius = target.radius  # in km

    # distance = radius / tan(pixels * ifov / 2)
    distance = target_radius / np.tan(n_pixels * ifov / 2)

    return float(distance)


def angular_size_to_distance(
    observer: SpiceInstrument | SpiceSpacecraft,
    target: SpiceBody,
    angular_size_rad: float,
) -> float:
    """Convert angular size to distance threshold.

    Inverse of the AngularSize property calculation.

    Formula:
        angular_size = 2 * arctan(target_radius / distance)
        distance = target_radius / tan(angular_size / 2)

    Args:
        observer: SpiceInstrument or SpiceSpacecraft
        target: SpiceBody target
        angular_size_rad: Angular size in radians

    Returns:
        Distance in km where target has given angular size
    """
    observer = as_spice_ref(observer)
    target = as_spice_ref(target)

    target_radius = target.radius  # in km

    # distance = radius / tan(angular_size / 2)
    distance = target_radius / np.tan(angular_size_rad / 2)

    return float(distance)
