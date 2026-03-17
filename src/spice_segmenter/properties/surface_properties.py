from enum import Enum

import numpy as np
import spiceypy
from attrs import define, field
from numpy.typing import ArrayLike
from planetary_coverage import SpiceRef, et
from planetary_coverage.spice import SpiceBody

from ..core.property import Property, PropertyTypes
from ..support.time_types import TIMES_TYPES


@define
class SurfaceIlluminationAngles(Property):
    """Computes illumination angles for points on the surface of a target body

    return phase, solar incidence, and emission"""
    
    _name = "illumination_angles"
    _unit = ["rad", "rad", "rad"]
    _type = PropertyTypes.VECTOR

    target: SpiceBody = field(converter=SpiceBody)
    surface_points: ArrayLike = field(converter=np.atleast_2d)
    observer: SpiceRef = field(converter=SpiceRef)
    frame: str = field(init=False)

    method: str = field(default="ELLIPSOID", kw_only=True)
    abcorr: str = field(default="NONE", kw_only=True)

    def __attrs_post_init__(self):
        self.frame = self.target.frame.name

    # method     I   Computation method.
    # target     I   Name of target body.
    # et         I   Epoch in TDB seconds past J2000 TDB.
    # fixref     I   Body-fixed, body-centered target body frame.
    # abcorr     I   Aberration correction flag.
    # obsrvr     I   Name of observing body.
    # spoint     I   Body-fixed coordinates of a target surface point.
    # @vectorize(signature="(),()->(n)")
    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum:
        return np.array(
            [
                spiceypy.ilumin(
                    method=self.method,
                    target=self.target.name,
                    et=et(time),
                    fixref=self.frame,
                    abcorr=self.abcorr,
                    obsrvr=self.observer.name,
                    spoint=p,
                )[2:]
                for p in self.surface_points
            ],
        )
