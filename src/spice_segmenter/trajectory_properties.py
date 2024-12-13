from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Union

import numpy as np
import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define, field
from loguru import logger as log
from planetary_coverage.spice import (
    SpiceBody,
    SpiceInstrument,
    SpiceRef,
    SpiceSpacecraft,
)
from planetary_coverage.spice.toolbox import groundtrack_velocity, sc_state

from spice_segmenter.property_base import Property, PropertyTypes
from spice_segmenter.types import TIMES_TYPES

from planetary_coverage.spice.toolbox import illum_angles
from spice_segmenter.component_selector import ComponentSelector
from spice_segmenter.utils import as_spice_ref

from .decorators import declare, vectorize
from .utils import et




PROPERTIES_REGISTRY = []


class MinMaxConditionTypes(Enum):
    LOCAL_MINIMUM = "local_minimum"
    LOCAL_MAXIMUM = "local_maximum"
    GLOBAL_MINIMUM = "global_minimum"
    GLOBAL_MAXIMUM = "global_maximum"



@define(repr=False, order=False, eq=False)
class TargetedPropertyMixin():
    observer: SpiceInstrument | SpiceSpacecraft = field(converter=as_spice_ref)
    target: SpiceBody = field(converter=as_spice_ref)
    light_time_correction: str = field(default="NONE", kw_only=True)

    def config(self, config: dict) -> None:
        log.debug("targeted property config here with instnace of {}", self.__class__.__name__)
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
class TargetedProperty( TargetedPropertyMixin, Property):
    pass

@declare(name="phase_angle", unit=pint.Unit("rad"))
class PhaseAngle(TargetedProperty):
    third_body: SpiceRef = field(factory=lambda: as_spice_ref("SUN"), converter=as_spice_ref)

    def __repr__(self) -> str:
        return f"Phase Angle of {self.target} with respect to {self.third_body} as seen from {self.observer}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return spiceypy.phaseq(  # type: ignore
            et(time),
            self.target.name,
            self.third_body.name,
            self.observer.name,
            self.light_time_correction,
        )

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["third_body"] = self.third_body.name
        config["property"] = self.name


@declare(name="distance", unit=pint.Unit("km"))
class Distance(TargetedProperty):
    def __repr__(self) -> str:
        return f"Distance of {self.target} from {self.observer}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return spiceypy.vnorm(  # type: ignore
            spiceypy.spkpos(
                self.target.name,
                et(time),
                self.observer.frame.name,
                self.light_time_correction,
                self.observer.name,
            )[0],
        )

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(name="sub_sc_velocity", unit=pint.Unit("km/s"))
class SubObserverPointVelocity(TargetedProperty):
    # method = field(default=SubObserverPointMethods.INTERCEPT)

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> np.ndarray:
        state = sc_state(time, self.observer.spacecraft, self.target, self.light_time_correction)
        return groundtrack_velocity(self.target, state)

    def __repr__(self) -> str:
        return f"Velocity of sub observer ({self.observer}) point on {self.target} surface."


@declare(name="angular_size", unit=pint.Unit("rad"))
class AngularSize(TargetedProperty):
    def __repr__(self) -> str:
        return f"Angular size of {self.target}, seen from {self.observer}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        d = Distance.__call__(self, time)
        return 2 * np.arctan(self.target.radius / d)  # type: ignore

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(name="sub_observer_pixel_scale", unit=pint.Unit("km/px"))
class SubObserverPixelScale(TargetedProperty):
    def __repr__(self) -> str:
        return f"Resultion of {self.target}, at the sub-{self.observer} point."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        from planetary_coverage.spice.toolbox import pixel_scale

        distance = Distance.__call__(self, time)
        return pixel_scale(inst=self.observer, target=self.target, emi=0, dist=distance)

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(name="approx_altitude", unit=pint.Unit("km"))
class ApproximatedAltitude(TargetedProperty):
    def __repr__(self) -> str:
        return (
            f"Approximated (distance-radius) altitude of {self.observer} over {self.target} surface (from sub-sc point)"
        )

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        distance = Distance.__call__(self, time)
        radius = self.target.radius

        return distance - radius

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(name="target_size_on_sensor", unit=pint.Unit("px"))
class TargetSizeOnSensor(TargetedProperty):
    def __repr__(self) -> str:
        return f"Size in pixels of {self.target}, on the {self.observer} sensor."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        angular_size = AngularSize.__call__(self, time)
        ifov = np.mean(self.observer.ifov)
        return angular_size / ifov

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(name="distance_in_target_radii", unit=pint.Unit("dimensionless"))
class DistanceInTargetBodyRadii(TargetedProperty):
    def __repr__(self) -> str:
        return f"Distance of {self.target}, from {self.observer} sensor, in {self.target} radii."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        distance = Distance.__call__(self, time)
        return distance / self.target.radius

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["property"] = self.name


@declare(
    name="sub_observer_illumination_angles",
    unit=[pint.Unit("deg"), pint.Unit("deg"), pint.Unit("deg")],
    property_type=PropertyTypes.VECTOR,
)
class SubObserverIlluminationAngles(TargetedProperty):
    def __repr__(self) -> str:
        return f"Sub {self.observer} point illumination angles on {self.target}"

    @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> float:
        from spice_segmenter.coordinates import SubObserverPoint

        pt = SubObserverPoint(self.observer, self.target)(time)

        return illum_angles(
            time,
            self.observer.spacecraft,
            self.target,
            pt,
            abcorr=self.light_time_correction,
            method="ELLIPSOID",
        )

    @property
    def incidence(self) -> Property:
        return ComponentSelector(self, 0, "incidence")

    @property
    def emission(self) -> Property:
        return ComponentSelector(self, 1, "emission")

    @property
    def phase(self) -> Property:
        return ComponentSelector(self, 2, "phase")


@declare(name="sub_obs_point_in_daylight", unit=pint.Unit("dimensionless"), property_type=PropertyTypes.BOOLEAN)
class SubObserverIsInDaylight(TargetedProperty):
    def __repr__(self) -> str:
        return f"Sub {self.observer} point on {self.target} is in daylight"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        i, e, p = SubObserverIlluminationAngles(self.observer, self.target)(time)
        return i < 90.0
