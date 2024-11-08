from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import planetary_coverage
from attrs import define, field

from spice_segmenter.spice_window import SpiceWindow
from spice_segmenter.visibility import BodyFOVVisibility


@define
class PointEvent:
    time: pd.Timestamp = field()
    description: str = field(default="")


def fov_in_out(
    target: str, start: str, end: str
) -> list[tuple[PointEvent, PointEvent]]:
    fov_vis = BodyFOVVisibility("JUICE_JANUS", target) == True
    w = SpiceWindow()
    w.add_interval(start, end)
    out = fov_vis.solve(w)

    output = []
    for i, item in out.to_pandas().iterrows():
        window_start, window_end = item

        in_event = PointEvent(
            time=window_start.to_numpy(), description=f"{target} ingresses fov"
        )
        out_event = PointEvent(
            time=window_end.to_numpy(), description=f"{target} egresses fov"
        )
        output.append((in_event, out_event))

    return output


def boolean_series_flips(
    times: Iterable[np.timedelta64], boolean_series: Iterable[bool]
) -> list[tuple[np.datetime64, Literal["TO_TRUE", "TO_FALSE"]]]:
    "determines times of flipping and polarity of the boolean series"
    _times: np.ndarray = np.array(times)

    flips = np.diff(np.asarray(boolean_series).astype(int))

    flip_ids = np.argwhere(flips).T[0]
    # to_day = np.argwhere(np.diff(is_day) == 1).T[0]

    out = []
    for fid in flip_ids:
        flip_time = _times[fid] + (_times[fid + 1] - _times[fid]) / 2
        flip_type = "TO_FALSE" if flips[fid] == -1 else "TO_TRUE"

        out.append((flip_time, flip_type))

    return out


def enters_exists_daylight(trajectory: Any) -> list[PointEvent]:
    """If it is a SC trajectory it will give you ingress and egress of sub-SC point in daylight.
    if it is an insturment trajectory it will give you the same but for the"""
    items = boolean_series_flips(trajectory.utc, trajectory.day)

    is_sc_traj = isinstance(
        trajectory, planetary_coverage.trajectory.SpacecraftTrajectory
    )

    print(f"Is a SC traj {is_sc_traj}")
    target = trajectory.target

    if is_sc_traj:
        pt_type = f"{trajectory.spacecraft} sub-point on {target}"
    else:
        pt_type = f"{trajectory.observer} boresight intercept point on {target}"

    events = []
    for t, type in items:
        if type == "TO_TRUE":
            ev = PointEvent(t, f"{pt_type} moves into light")
        else:
            ev = PointEvent(t, f"{pt_type} moves into dark")

        events.append(ev)

    return events
