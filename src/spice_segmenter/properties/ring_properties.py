from __future__ import annotations

import numpy as np
import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define, field
from planetary_coverage.spice import SpiceBody, SpiceObserver, et

# from jana.toolbox import compute_sun_vector
from planetary_coverage.spice.toolbox import sun_pos

from ..core.property import BooleanProperty, PropertyTypes
from ..support.decorators import vectorize
from ..support.spice_utilities import et
from ..support.time_types import TIMES_TYPES


def ring_ansae_phase_angles(
    time, observer="JUICE", pts_radiuses=np.array([-250000, -100000, 100000, 250000]),
):
    """
    Return phase angles at specific points along the ring ansae (apparent extremities).
    
    Computes phase angles at the visible extremities of Jupiter's ring system - the ansae
    are the easternmost and westernmost apparent points where the line of sight is tangent
    to the rings.

    Radii values from https://juicesoc.esac.esa.int/help/81_science_models.html
    
    Args:
        time: Single time or array of times
        observer: Observer name (default: "JUICE")
        pts_radiuses: Array of radii at which to compute phase angles along the ansae
        
    Returns:
        If time is scalar: array of phase angles at each radius point
        If time is array: 2D array where each row is angles for one time
    """
    observer = SpiceObserver(observer)
    jup = SpiceBody("jupiter")
    
    # Convert time to ET
    time_et = et(time)
    
    # Check if time is scalar or array
    is_scalar = np.ndim(time_et) == 0
    if is_scalar:
        time_et = np.array([time_et])
    
    # Process each time
    all_angles = []
    for t in time_et:
        sun = sun_pos(t, jup)  # sun pos relative to jupiter

        sc_pos = spiceypy.spkpos(observer.name, t, jup.frame.name, "NONE", jup.name)[
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
        
        all_angles.append(angles)
    
    # Return appropriate shape
    if is_scalar:
        return all_angles[0]
    else:
        return np.array(all_angles)


def min_max_ring_ansae_phase_angle(time, observer="JUICE"):
    """Get min/max phase angles for left and right ring ansae.
    
    Returns:
        tuple: (minR, maxR, minL, maxL) where R = right ansa (negative radii), L = left ansa (positive radii)
               If time is array, returns tuple of arrays
    """
    angles = ring_ansae_phase_angles(
        time,
        observer=observer,
        pts_radiuses=np.array([-250000, -100000, 100000, 250000]),
    )
    
    # Handle both scalar and array cases
    if angles.ndim == 1:
        # Single time: angles is 1D array of 4 values
        return np.min(angles[:2]), np.max(angles[:2]), np.min(angles[2:]), np.max(angles[2:])
    else:
        # Multiple times: angles is 2D array (n_times, 4)
        minR = np.min(angles[:, :2], axis=1)
        maxR = np.max(angles[:, :2], axis=1)
        minL = np.min(angles[:, 2:], axis=1)
        maxL = np.max(angles[:, 2:], axis=1)
        return minR, maxR, minL, maxL


def is_ring_ansae_phase_angles_lower_than(time, angle):
    minR, maxR, minL, maxL = min_max_ring_ansae_phase_angle(time)
    return (minR < angle) | (minL < angle)

def is_ring_ansae_phase_angles_greater_than(time, angle):
    minR, maxR, minL, maxL = min_max_ring_ansae_phase_angle(time)
    return (maxR > angle) | (maxL > angle)


def is_ring_ansae_phase_angles_in_between(time, min_angle, max_angle):
    minR, maxR, minL, maxL = min_max_ring_ansae_phase_angle(time)
    return ((minR > min_angle ) and (maxR < max_angle)) | ((minL > min_angle) and (maxL < max_angle))

@define(repr=False, order=False, eq=False)
class RingAnsaePhaseLowerThan(BooleanProperty):
    _name = "ring_ansae_phase_lower_than"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN
    
    value_deg: float = field()
    observer: SpiceObserver = field(default="JUICE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle at the ring ansae are lower than {self.value_deg}° at either ansa of Jupiter's rings, as seen by {self.observer}."


@define(repr=False, order=False, eq=False)
class RingAnsaePhaseGreaterThan(BooleanProperty):
    _name = "ring_ansae_phase_greater_than"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN
    
    value_deg: float = field()
    observer: SpiceObserver = field(default="JUICE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle at the ring ansae are greater than {self.value_deg}° at either ansa of Jupiter's rings, as seen by {self.observer}."



@define(repr=False, order=False, eq=False)
class RingAnsaePhaseWithinRange(BooleanProperty):
    _name = "ring_ansae_phase_within_range"
    _unit = pint.Unit("dimensionless")
    _type = PropertyTypes.BOOLEAN
    
    lower_deg: float = field()
    upper_deg: float = field()
    observer: SpiceObserver = field(default="JUICE_JANUS", converter = SpiceObserver)

    def __repr__(self) -> str:
        return f"The phase angle at the ring ansae is completely within [{self.lower_deg}°, {self.upper_deg}°] at either ansa of Jupiter's rings, as seen by {self.observer}."

