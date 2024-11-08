"""Quick-use functions to query spice_segmenter."""

import pandas as pd

from spice_segmenter import SpiceWindow, config
from spice_segmenter.visibility import BodyFOVVisibility


def find_visibility_intervals(
    target, start, end, observer="juice_janus", solver_step_sec=5
) -> pd.DataFrame:
    """Find the first interval in which the target is visibile in the FOV of the observer."""
    config.solver_step = solver_step_sec

    v = BodyFOVVisibility(observer, target)
    w = SpiceWindow.from_start_end(start, end)

    interval = (v == True).solve(w)
    return interval.to_pandas()


def find_first_visibility_interval(
    target, start, end, observer="juice_janus", solver_step_sec=5
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
