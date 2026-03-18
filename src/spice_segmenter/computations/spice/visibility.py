"""SPICE compute functions for visibility properties."""

from __future__ import annotations

import numpy as np
import spiceypy

from ...properties.visibility_properties import BodyFOVVisibility


# ---------------------------------------------------------------------------
# BodyFOVVisibility
# ---------------------------------------------------------------------------


def fov_visibility_scalar(prop: BodyFOVVisibility, time_et: float) -> bool:
    return spiceypy.fovtrg(
        prop.observer.name,
        prop.target.name,
        "ELLIPSOID",
        prop.target.frame,
        prop.light_time_correction,
        prop.observer.name,
        time_et,
    )


def fov_visibility_vector(prop: BodyFOVVisibility, times_et: np.ndarray) -> np.ndarray:
    from spiceypy import cyice

    return cyice.fovtrg_v(
        prop.observer.name,
        prop.target.name,
        "ELLIPSOID",
        prop.target.frame,
        prop.light_time_correction,
        prop.observer.name,
        times_et,
    )
