from enum import Enum

import numpy as np
import spiceypy
from planetary_coverage import et
from pydantic import Field

from spice_segmenter.support.model import segmenter_dataclass
from spice_segmenter.support.types import SpiceBodyField, SpiceRefField, converted

from ..core.property import Property, PropertyTypes
from ..support.time_types import TIMES_TYPES

_SurfacePoints = converted(
    np.atleast_2d,
    serializer=lambda a: np.asarray(a).tolist(),
    json_schema={"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
)


@segmenter_dataclass
class SurfaceIlluminationAngles(Property):
    """Computes illumination angles for points on the surface of a target body

    return phase, solar incidence, and emission"""

    _name = "illumination_angles"
    _unit = ["rad", "rad", "rad"]
    _type = PropertyTypes.VECTOR

    target: SpiceBodyField
    surface_points: _SurfacePoints
    observer: SpiceRefField
    frame: str = Field(default="", init=False)

    method: str = Field(default="ELLIPSOID", kw_only=True)
    abcorr: str = Field(default="NONE", kw_only=True)

    def __post_init__(self):
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
