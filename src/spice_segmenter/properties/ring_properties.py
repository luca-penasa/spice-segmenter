from __future__ import annotations

import numpy as np
import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import field
from planetary_coverage.spice import SpiceBody, SpiceObserver, et

# from jana.toolbox import compute_sun_vector
from planetary_coverage.spice.toolbox import sun_pos

from ..core.property import BooleanProperty, PropertyTypes
from ..support.decorators import declare, vectorize
from ..support.spice_utilities import et
from ..support.time_types import TIMES_TYPES


def ring_plane_angles(
    time, observer="JUICE", pts_radiuses=np.array([-250000, -100000, 100000, 250000]),
):
    """
    Return phase angles at specific points on the ring plane.

    radiuses values from https://juicesoc.esac.esa.int/help/81_science_models.html
    """
    observer = SpiceObserver(observer)
    jup = SpiceBody("jupiter")
    time = et(time)
    sun = sun_pos( time, jup)  # sun pos relative to jupiter

    sc_pos = spiceypy.spkpos(observer.name, time, jup.frame.name, "NONE", jup.name)[
        0
    ]  # sc position wrt jupiter center
    # compute the position of interest points on the ring plane

    planes_intersections = np.cross(
        sc_pos, [0, 0, 1],
    )  # intersection of the ring plane and the plane perpendicular to the vector sc->jup
    planes_intersections /= np.linalg.norm(planes_intersections)
    ring_pts = np.array([planes_intersections * pt for pt in pts_radiuses])

    rsc = sc_pos - ring_pts  # sc position relative to the consider ring points.
    rsc = rsc / np.linalg.norm(rsc, axis=1, keepdims=True)

    rsun = sun - ring_pts  # sun position relative to the consider ring points.
    rsun = rsun / np.linalg.norm(rsun, axis=1, keepdims=True)

    angles = np.rad2deg(
        np.arccos(np.einsum("ij,ij->i", rsc, rsun)),
    )  # computing phase angles at that location.

    return angles


def min_max_ring_plane_phase_angle(time, observer="JUICE"):
    angles = ring_plane_angles(
        time,
        observer=observer,
        pts_radiuses=np.array([-250000, -100000, 100000, 250000]),
    )

    return np.min(angles[:2]), np.max(angles[:2]), np.min(angles[2:]), np.max(angles[2:])


def is_rings_phase_angles_lower_than(time, angle):
    minR, maxR, minL, maxL = min_max_ring_plane_phase_angle(time)
    return (minR < angle) | (minL < angle)

def is_rings_phase_angles_greater_than(time, angle):
    minR, maxR, minL, maxL = min_max_ring_plane_phase_angle(time)
    return (maxR > angle) | (maxL > angle)


def is_rings_phase_angles_in_betweeen(time, min_angle, max_angle):
    minR, maxR, minL, maxL = min_max_ring_plane_phase_angle(time)
    return ((minR > min_angle ) and (maxR < max_angle)) | ((minL > min_angle) and (maxL < max_angle))

@declare(name="ring_system_phases_lower_than", unit=pint.Unit("dimensionless"), property_type=PropertyTypes.BOOLEAN)
class RingSystemPhaseLowerThan(BooleanProperty):
    value_deg: float = field()
    observer: SpiceObserver = field(default="JUICCE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle of the ring system are lower than {self.value_deg} either at right or left of Jupiter, as seen by {self.observer}."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return is_rings_phase_angles_lower_than(time, self.value_deg)


@declare(name="ring_system_phases_greater_than", unit=pint.Unit("dimensionless"), property_type=PropertyTypes.BOOLEAN)
class RingSystemPhaseGreaterThan(BooleanProperty):
    value_deg: float = field()
    observer: SpiceObserver = field(default="JUICE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle of the ring system are greater than {self.value_deg} either at right or left of Jupiter, as seen by {self.observer}."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return is_rings_phase_angles_greater_than(time, self.value_deg)




@declare(name="ring_system_phases_within_range", unit=pint.Unit("dimensionless"), property_type=PropertyTypes.BOOLEAN)
class RingSystemPhaseWithinRange(BooleanProperty):
    lower_deg: float = field()
    upper_deg: float = field()
    observer: SpiceObserver = field(default="JUICE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle of the ring system is completely in the bound [{self.lower_deg}, {self.upper_deg}] either at right or left of Jupiter, as seen by {self.observer}."

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return is_rings_phase_angles_in_betweeen(time, self.lower_deg, self.upper_deg)

