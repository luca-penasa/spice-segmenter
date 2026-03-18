"""Event-building helpers that produce :class:`~spice_segmenter.core.TimeSegment`
point events (zero-duration segments) and :class:`~spice_segmenter.core.TimeSegmentsCollection`
collections of such events.
"""

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import planetary_coverage

from ..collections.quick_access import find_ca
from ..core.time_segment import TimeSegment
from ..core.time_segments_collection import TimeSegmentsCollection
from ..properties.visibility_properties import BodyFOVVisibility


def fov_in_out(
    target: str,
    start: str,
    end: str,
) -> list[tuple[TimeSegment, TimeSegment]]:
    """Return ingress / egress point pairs for *target* entering the JANUS FOV.

    Each element of the returned list is a ``(ingress, egress)`` pair of
    zero-duration :class:`~spice_segmenter.core.TimeSegment` instances.
    """
    fov_vis = BodyFOVVisibility("JUICE_JANUS", target) == True
    w = TimeSegmentsCollection.from_start_end(start, end)
    out = fov_vis.solve(w)

    output = []
    for seg in out:
        in_event = TimeSegment.at_time(
            seg.start,
            label=f"{target} ingresses fov",
            metadata={"target": target, "event_type": "FOV_INGRESS"},
        )
        out_event = TimeSegment.at_time(
            seg.end,
            label=f"{target} egresses fov",
            metadata={"target": target, "event_type": "FOV_EGRESS"},
        )
        output.append((in_event, out_event))

    return output


def boolean_series_flips(
    times: Iterable[np.timedelta64], boolean_series: Iterable[bool],
) -> list[tuple[np.datetime64, Literal["TO_TRUE", "TO_FALSE"]]]:
    """Determine transition times and polarity of a boolean time series."""
    _times: np.ndarray = np.array(times)
    flips = np.diff(np.asarray(boolean_series).astype(int))
    flip_ids = np.argwhere(flips).T[0]

    out = []
    for fid in flip_ids:
        flip_time = _times[fid] + (_times[fid + 1] - _times[fid]) / 2
        flip_type = "TO_FALSE" if flips[fid] == -1 else "TO_TRUE"
        out.append((flip_time, flip_type))
    return out


def enters_exits_daylight(trajectory: Any) -> list[TimeSegment]:
    """Return point events for sub-point daylight ingress / egress transitions.

    Works for both spacecraft trajectories (sub-SC point) and instrument
    trajectories (boresight intercept point).  Returns a list of zero-duration
    :class:`~spice_segmenter.core.TimeSegment` instances.
    """
    items = boolean_series_flips(trajectory.utc, trajectory.day)

    is_sc_traj = isinstance(
        trajectory, planetary_coverage.trajectory.SpacecraftTrajectory,
    )
    target = trajectory.target

    if is_sc_traj:
        pt_type = f"{trajectory.spacecraft} sub-point on {target}"
    else:
        pt_type = f"{trajectory.observer} boresight intercept point on {target}"

    events: list[TimeSegment] = []
    for t, kind in items:
        if kind == "TO_TRUE":
            ev = TimeSegment.at_time(
                t,
                label=f"{pt_type} moves into light",
                metadata={"event_type": "INGRESS_DAYLIGHT", "target": target},
            )
        else:
            ev = TimeSegment.at_time(
                t,
                label=f"{pt_type} moves into dark",
                metadata={"event_type": "EGRESS_DAYLIGHT", "target": target},
            )
        events.append(ev)
    return events


# Backwards-compatible alias (old spelling had a typo: "exists")
enters_exists_daylight = enters_exits_daylight


def create_ca_deltas(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    targets: list[str],
    time_offsets: list[str] | None = None,
    abbreviations: dict[str, str] | None = None,
    excluded_offsets: dict[str, list[str]] | None = None,
) -> TimeSegmentsCollection:
    """Create point events for multiple targets based on their closest approaches.

    Returns a :class:`~spice_segmenter.core.TimeSegmentsCollection` of
    zero-duration :class:`~spice_segmenter.core.TimeSegment` instances,
    sorted by time.  Each segment carries a ``label`` (e.g. ``"GANYMEDE_CA"``)
    and metadata with ``target``, ``event_type``, and (for offsets)
    ``offset`` and ``ca_time``.

    Parameters
    ----------
    start_time, end_time:
        Search bounds for the closest-approach finder.
    targets:
        SPICE body names (e.g. ``["GANYMEDE", "JUPITER"]``).
    time_offsets:
        Offset strings relative to each CA (e.g. ``["-24 h", "+1 h"]``).
        Defaults to a standard ±24 h ladder.
    abbreviations:
        Short labels per target used as event-name prefixes.
    excluded_offsets:
        Per-target offsets to skip.
    """
    if time_offsets is None:
        time_offsets = [
            "-24 h", "-12 h", "-6 h", "-2 h", "-1 h", "-30 m",
            "+30 m", "+1 h", "+2 h", "+6 h", "+12 h", "+24 h",
        ]
    if abbreviations is None:
        abbreviations = {
            "JUPITER": "J", "GANYMEDE": "G", "CALLISTO": "C", "EUROPA": "E",
        }
    if excluded_offsets is None:
        excluded_offsets = {}

    segments: list[TimeSegment] = []
    ca_segments: dict[str, TimeSegment] = {}

    for target in targets:
        ca_seg = find_ca(start_time, end_time, target)
        ca_name = f"{target}_CA"
        ca_event = TimeSegment.at_time(
            ca_seg.start,
            label=ca_name,
            value=ca_seg.value,
            property_name="distance",
            metadata={"target": target, "event_type": "CA"},
        )
        segments.append(ca_event)
        ca_segments[ca_name] = ca_event

    for offset_str in time_offsets:
        offset = pd.Timedelta(offset_str)
        for ca_name, ca_seg in ca_segments.items():
            target_name = ca_name.replace("_CA", "")
            if target_name in excluded_offsets and offset_str in excluded_offsets[target_name]:
                continue
            prefix = abbreviations.get(target_name, target_name)
            segments.append(
                TimeSegment.at_time(
                    ca_seg.start + offset,
                    label=f"{prefix}{offset_str}",
                    metadata={
                        "target": target_name,
                        "event_type": "CA_OFFSET",
                        "offset": offset_str,
                        "ca_time": str(ca_seg.start),
                    },
                )
            )

    segments.sort(key=lambda s: s.start)
    return TimeSegmentsCollection(segments=segments)
