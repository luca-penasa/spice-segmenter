"""Quick-use functions to query spice_segmenter."""

import pandas as pd

from spice_segmenter import SpiceWindow, config
from spice_segmenter.ops import MinMaxConditionTypes, MinMaxConstraint
from spice_segmenter.trajectory_properties import Distance
from spice_segmenter.visibility import BodyFOVVisibility

from planetary_coverage.spice import SpiceBody

def find_visibility_intervals(target, start, end, observer="juice_janus", solver_step_sec=5) -> pd.DataFrame:
    """Find the first interval in which the target is visibile in the FOV of the observer."""
    config.solver_step = solver_step_sec

    v = BodyFOVVisibility(observer, target)
    w = SpiceWindow.from_start_end(start, end)

    interval = (v == True).solve(w)
    return interval.to_pandas()


def body_radii_to_distance(n_radii: float, target: str = "JUPITER"):
    """Return the distance to the body corresponding to the given number of radii of the body."""
    target = SpiceBody(target)
    return n_radii * target.radius

def find_first_visibility_interval(
    target,
    start,
    end,
    observer="juice_janus",
    solver_step_sec=5,
) -> (pd.Timestamp, pd.Timestamp):
    item = find_visibility_intervals(
        target=target,
        start=start,
        end=end,
        observer=observer,
        solver_step_sec=solver_step_sec,
    ).iloc[0]

    s = item.start
    e = item.end
    return s, e


def find_ca(start, end, target="JUPITER", observer="JUICE_JANUS") -> pd.Timestamp:
    """Returns the closest approach within start and end."""
    dist = Distance(observer, target)

    c = MinMaxConstraint(dist, MinMaxConditionTypes.GLOBAL_MINIMUM)
    w = SpiceWindow.from_start_end(start, end)

    ca = c.solve(w)
    if not ca.is_point():
        raise ValueError("Could not determine CA.")

    return ca.to_start_end()[0]
