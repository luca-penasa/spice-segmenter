"""SPICE compute functions for the Occultation property."""

from __future__ import annotations

import numpy as np
import spiceypy

from ...properties.occultation_types import Occultation, OccultationTypes


def _remap_to_enum(value: int) -> OccultationTypes:
    if value == -3:
        return OccultationTypes.FULL
    if value == -2:
        return OccultationTypes.ANNULAR
    if value == -1:
        return OccultationTypes.PARTIAL
    return OccultationTypes.NONE


def occultation_scalar(prop: Occultation, time_et: float) -> OccultationTypes:
    v = spiceypy.occult(
        prop.back.name,
        "ELLIPSOID",
        prop.back.frame,
        prop.front.name,
        "ELLIPSOID",
        prop.front.frame,
        prop.light_time_correction,
        prop.observer.name,
        time_et,
    )
    return _remap_to_enum(v)


def occultation_vector(prop: Occultation, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice

    raw = cyice.occult_v(
        prop.back.name,
        "ELLIPSOID",
        prop.back.frame,
        prop.front.name,
        "ELLIPSOID",
        prop.front.frame,
        prop.light_time_correction,
        prop.observer.name,
        times_et,
    )
    return np.vectorize(_remap_to_enum)(raw)
