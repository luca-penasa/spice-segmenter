"""SPICE compute functions for reflector (Jupiter-shine) properties."""

from __future__ import annotations

import numpy as np

from ...properties.reflector_properties import (
    JupiterRise,
    JupiterRiseRatio,
    ShineProperties,
    relfected_light_properties,
)


def shine_properties_scalar(prop: ShineProperties, time_et: float) -> np.ndarray:
    return relfected_light_properties(
        time_et,
        target_name=prop.target,
        observer=prop.observer,
        reflector=prop.reflector,
        light_source=prop.light_source,
        abcorr=prop.light_time_correction,
    )


def jupiter_rise_scalar(prop: JupiterRise, time_et: float) -> bool:
    result = relfected_light_properties(
        time_et,
        target_name=prop.target,
        observer=prop.observer,
        reflector="JUPITER",
        abcorr=prop.light_time_correction,
    )
    return bool(result[0] > result[3])


def jupiter_rise_ratio_scalar(prop: JupiterRiseRatio, time_et: float) -> float:
    result = relfected_light_properties(
        time_et,
        target_name=prop.target,
        observer=prop.observer,
        reflector="JUPITER",
        abcorr=prop.light_time_correction,
    )
    return float(result[0] / result[3])
