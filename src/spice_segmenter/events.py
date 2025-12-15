from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import planetary_coverage
from attrs import define, field

from spice_segmenter.quick import find_ca
from spice_segmenter.spice_window import SpiceWindow
from spice_segmenter.visibility import BodyFOVVisibility


@define
class PointEvent:
    time: pd.Timestamp = field()
    description: str = field(default="")
    metadata: dict[str, Any] = field(factory=dict)


def fov_in_out(
    target: str, start: str, end: str,
) -> list[tuple[PointEvent, PointEvent]]:
    fov_vis = BodyFOVVisibility("JUICE_JANUS", target) == True
    w = SpiceWindow()
    w.add_interval(start, end)
    out = fov_vis.solve(w)

    output = []
    for i, item in out.to_pandas().iterrows():
        window_start, window_end = item

        in_event = PointEvent(
            time=window_start.to_numpy(), description=f"{target} ingresses fov",
        )
        out_event = PointEvent(
            time=window_end.to_numpy(), description=f"{target} egresses fov",
        )
        output.append((in_event, out_event))

    return output


def boolean_series_flips(
    times: Iterable[np.timedelta64], boolean_series: Iterable[bool],
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
        trajectory, planetary_coverage.trajectory.SpacecraftTrajectory,
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
            ev.metadata["type"] = "INGRESS_DAYLIGHT"
        else:
            ev = PointEvent(t, f"{pt_type} moves into dark")
            ev.metadata["type"] = "EGRESS_DAYLIGHT"

        events.append(ev)

    return events



def create_ca_deltas(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    targets: list[str],
    time_offsets: list[str] | None = None,
    abbreviations: dict[str, str] | None = None,
    excluded_offsets: dict[str, list[str]] | None = None,
) -> dict[str, PointEvent]:
    """
    Create point events for multiple targets based on their closest approaches.
    
    Parameters
    ----------
    start_time : pd.Timestamp
        Start time for finding closest approaches
    end_time : pd.Timestamp
        End time for finding closest approaches
    targets : list[str]
        List of target body names (e.g., ["GANYMEDE", "JUPITER"])
    time_offsets : list[str], optional
        List of time offset strings (e.g., ["-24 h", "+1 h", "-30 m"])
        Default: ["-24 h", "-12 h", "-6 h", "-2 h", "-1 h", "-30 m", 
                  "+30 m", "+1 h", "+2 h", "+6 h", "+12 h", "+24 h"]
    abbreviations : dict[str, str], optional
        Mapping of target names to abbreviations for offset events
        (e.g., {"GANYMEDE": "G", "JUPITER": "J"})
    excluded_offsets : dict[str, list[str]], optional
        Mapping of target names to lists of offsets to exclude
        (e.g., {"JUPITER": ["-1 h", "+1 h"]})
    
    Returns
    -------
    dict[str, PointEvent]
        Dictionary mapping event names to PointEvent objects
    """
    if time_offsets is None:
        time_offsets = [
            "-24 h", "-12 h", "-6 h", "-2 h", "-1 h", "-30 m",
            "+30 m", "+1 h", "+2 h", "+6 h", "+12 h", "+24 h",
        ]

    if abbreviations is None:
        abbreviations = {
            "JUPITER": "J",
            "GANYMEDE": "G",
            "CALLISTO": "C",
            "EUROPA": "E",
        }

    if excluded_offsets is None:
        excluded_offsets = {}

    # Find closest approaches for all targets
    point_events = {}
    ca_times = {}  # Store CA times for offset calculations

    for target in targets:
        ca_time = find_ca(start_time, end_time, target)
        ca_name = f"{target}_CA"

        ca_event = PointEvent(
            time=ca_time,
            description=f"{target} closest approach",
            metadata={"target": target, "event_type": "CA"},
        )
        point_events[ca_name] = ca_event
        ca_times[ca_name] = ca_time

    # Add offset events
    for offset_str in time_offsets:
        offset = pd.Timedelta(offset_str)

        for ca_name, ca_time in ca_times.items():
            target_name = ca_name.replace("_CA", "")

            # Check if this offset should be excluded for this target
            if (
                target_name in excluded_offsets
                and offset_str in excluded_offsets[target_name]
            ):
                continue

            # Use abbreviation if provided, otherwise use target name
            name_prefix = abbreviations.get(target_name, target_name)

            offset_event_name = f"{name_prefix}{offset_str}"
            offset_event_time = ca_time + offset

            offset_event = PointEvent(
                time=offset_event_time,
                description=f"{target_name} CA {offset_str}",
                metadata={
                    "target": target_name,
                    "event_type": "CA_OFFSET",
                    "offset": offset_str,
                    "ca_time": ca_time,
                },
            )
            point_events[offset_event_name] = offset_event

    return point_events
