""" "Properties related to the illumination condition of a target subject to illumination by a 'reflector'. Aka 'shine'.
Adapted from the original implementation by Klaus-Dieter Matz (DLR), end of 2024.
"""

import numpy as np
import pint
import spiceypy
from attrs import define, field
from planetary_coverage import et
from planetary_coverage.spice import SpiceBody

from ..core.property import BooleanProperty, Property, PropertyTypes
from ..properties.component_selector import ComponentSelector
from ..properties.observation_properties import (
    TargetedProperty,
    TargetedPropertyMixin,
)
from ..support.decorators import vectorize
from ..support.time_types import TIMES_TYPES


def relfected_light_properties(
    time, target_name="CALLISTO", observer="JUICE_JANUS", reflector="JUPITER", light_source="SUN", abcorr="LT+S",
):
    """Calculate 'Jupiter Shine' parameters."""

    time = et(time)
    target = SpiceBody(target_name)
    target_frame = target.frame.name
    reflector_radii = SpiceBody(reflector).radii

    reflector_pos, reflector2sub_sc_lt = spiceypy.spkpos(reflector, time, target_frame, abcorr, target_name)
    sub_sc_point, _, sc2sub_sc = spiceypy.subpnt(
        "NEARPOINT/ELLIPSOID", target_name, time, target_frame, abcorr, observer,
    )

    sub_sc2reflector = sub_sc_point + reflector_pos
    reflector2sun, _ = spiceypy.spkpos(light_source, time - reflector2sub_sc_lt, target_frame, abcorr, reflector)

    return np.array(
        [
            jupiter_elevation(sub_sc2reflector, sub_sc_point),
            jupiter_phase_angle(sub_sc2reflector, reflector2sun),
            pseudo_phase_angle(sc2sub_sc, sub_sc2reflector),
            jupiter_angular_radius(sub_sc2reflector, reflector_radii),
        ],
    )


def jupiter_elevation(sub_sc2reflector, sub_sc_point):
    """Calculate the elevation of the reflector above the horizon at the sub-s/c point."""
    return 90.0 - np.rad2deg(spiceypy.vsep(sub_sc2reflector, sub_sc_point))


def jupiter_phase_angle(sub_sc2reflector, reflector2sun):
    """Calculate the phase angle of the reflector as seen from the sub-s/c point."""
    return np.rad2deg(spiceypy.vsep(-sub_sc2reflector, reflector2sun))


def jupiter_angular_radius(sub_sc2reflector, reflector_radii):
    """Calculate the angular radius of the reflector as seen from the sub-s/c point."""
    return np.rad2deg(np.arctan(reflector_radii[0] / spiceypy.vnorm(sub_sc2reflector)))


def pseudo_phase_angle(sc2sub_sc, sub_sc2reflector):
    """Calculate the pseudo 'Phase' angle (reflector <- sub-s/c -> S/C)."""
    return np.rad2deg(spiceypy.vsep(-sc2sub_sc, sub_sc2reflector))


@define(repr=False, order=False, eq=False)
class ShineProperties(TargetedProperty):
    _name = "shine_properties"
    _unit = [pint.Unit("deg"), pint.Unit("deg"), pint.Unit("deg"), pint.Unit("deg")]
    _type = PropertyTypes.VECTOR
    
    reflector = field(converter=SpiceBody, kw_only=True)
    light_source = field(converter=SpiceBody, default="SUN", kw_only=True)

    def __repr__(self) -> str:
        return f"Shine properties for the sub observer point on {self.target}, illuminated by reflected light of {self.reflector}, as seen from {self.observer}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return relfected_light_properties(
            time,
            target_name=self.target,
            observer=self.observer,
            reflector=self.reflector,
            light_source=self.light_source,
            abcorr=self.light_time_correction,
        )

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["reflector"] = self.reflector.name
        config["property"] = self.name

    @property
    def reflector_elevation(self) -> Property:
        return ComponentSelector(self, 0, "reflector_elevation")

    @property
    def reflector_phase_angle(self) -> Property:
        return ComponentSelector(self, 1, "reflector_phase_angle")

    @property
    def local_phase_angle(self) -> Property:
        return ComponentSelector(self, 2, "local_phase_angle")

    @property
    def reflector_angular_radius(self) -> Property:
        return ComponentSelector(self, 3, "reflector_angular_radius")


@define(repr=False, order=False, eq=False)
class JupiterRise(TargetedPropertyMixin, BooleanProperty):
    _name = "jupiter_rise"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN
    
    def __repr__(self) -> str:
        return f"Jupiter rise status for the sub-{self.observer} {self.target}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        result = relfected_light_properties(
            time,
            target_name=self.target,
            observer=self.observer,
            reflector="JUPITER",
            abcorr=self.light_time_correction,
        )
        el = result[0]   # reflector_elevation
        r  = result[3]   # reflector_angular_radius
        return el > r


@define(repr=False, order=False, eq=False)
class JupiterRiseRatio(TargetedProperty):
    _name = "jupiter_rise_ratio"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.SCALAR
    
    def __repr__(self) -> str:
        return f"Jupiter rise ratio (elevation/apparent jupiter radius) for the sub-{self.observer} on {self.target}"

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        result = relfected_light_properties(
            time,
            target_name=self.target,
            observer=self.observer,
            reflector="JUPITER",
            abcorr=self.light_time_correction,
        )
        el = result[0]   # reflector_elevation
        r  = result[3]   # reflector_angular_radius
        return el / r


@define(repr=False, order=False, eq=False)
class JupiterShineIdealCondition(TargetedPropertyMixin, BooleanProperty):
    _name = "jupiter_shine_ideal_condition"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN
    
    max_apparent_jupiter_phase = field(default=90)  # half disk is illuminated
    min_rise_ratio = field(default=1)  # half above the sub-observer horizon

    def __repr__(self) -> str:
        return f"Sub-{self.observer} on {self.target} is in ideal condition for Jupiter shine (apparent phase of Jupiter < {self.max_apparent_jupiter_phase} and el/radius of jupiter > {self.min_rise_ratio})"

    def __call__(self, time: TIMES_TYPES) -> float:
        # Call the underlying relfected_light_properties directly without going through
        # the vectorized ShineProperties to avoid nested vectorization issues
        time_et = et(time)
        is_scalar = np.isscalar(time_et)

        result = relfected_light_properties(
            time_et,
            target_name=self.target,
            observer=self.observer,
            reflector="JUPITER",
            light_source="SUN",
            abcorr=self.light_time_correction,
        )

        # result is shape (4,) for scalar or (4, n) for array
        el = result[0]
        phase = result[1]
        angular_radius = result[3]

        condition = (el / angular_radius > self.min_rise_ratio) & (
            phase < self.max_apparent_jupiter_phase
        )

        # Return scalar boolean if input was scalar, otherwise return array
        return bool(condition) if is_scalar else condition
