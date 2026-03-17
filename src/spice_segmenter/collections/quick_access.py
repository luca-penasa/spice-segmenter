"""Quick-use functions to query spice_segmenter."""

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger as log
from planetary_coverage.spice import SpiceBody

from ..core.spice_window import SpiceWindow
from ..ops.constraint_operations import MinMaxConstraint
from ..properties.observation_properties import Distance, MinMaxConditionTypes
from ..properties.visibility_properties import BodyFOVVisibility
from ..support.config import config


def flybys_windows(
    target="callisto",
    start="2031-01-01T00:00:00.000",
    end="2034-10-30T22:30:44.625",
    observer="JUICE_JANUS",
    altitude_treshold_km=80000,
):
    d = Distance(observer, target)
    distance_treshold_km = altitude_treshold_km + SpiceBody(target).radius
    return (d < f"{distance_treshold_km} km").solve(
        SpiceWindow.from_start_end(start, end),
    )


def flybys_cas(
    target="callisto",
    start="2031-01-01T00:00:00.000",
    end="2034-10-30T22:30:44.625",
    observer="JUICE_JANUS",
    altitude_treshold_km=80000,
):
    windows = flybys_windows(target, start, end, observer, altitude_treshold_km)
    return [find_ca(*win.to_start_end(), target) for win in windows]


def find_visibility_intervals(
    target,
    start,
    end,
    observer="juice_janus",
    solver_step_sec=5,
) -> pd.DataFrame:
    """Find the first interval in which the target is visibile in the FOV of the observer."""
    config.solver_step = solver_step_sec

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

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


def find_apsis(
    start,
    end,
    target="JUPITER",
    observer="JUICE_JANUS",
    type: Literal["apo", "peri"] = "apo",
):
    """
    Find the point in times of the apo or peri apsis of a target body with respect to an observer.
    """

    if type not in ["apo", "peri"]:
        raise ValueError("type must be either 'apo' or 'peri'")

    if type == "apo":
        ctype = MinMaxConditionTypes.LOCAL_MAXIMUM
    elif type == "peri":
        ctype = MinMaxConditionTypes.LOCAL_MINIMUM

    target_distance = Distance(observer, target)
    w = SpiceWindow.from_start_end(start, end)
    c = MinMaxConstraint(target_distance, ctype)
    ca = c.solve(w)
    return ca.to_pandas()["start"]


def get_periapsis_interval_table(start, end, target="JUPITER", observer="JUICE_JANUS"):
    """
    Find periapsis of a target body with respect to an observer in the given time interval.
    This function returns a set of intervals for each periapsis event.
    Each interval is defined by the start and end time of the apoapsis event just before and after the periapsis event.
    If no apoapsis event is found before or after the considered periapsis event,
    the event start or end is created symmetrically with respect to the periapsis event.

    The function also returns a table with the periapsis and apoapsis events alone, including their distance to the target body.

    """

    target_distance = Distance(observer, target)

    apo = find_apsis(start, end, target, type="apo")
    peri = find_apsis(start, end, target, type="peri")

    apo_d = target_distance(apo)
    peri_d = target_distance(peri)

    apo_table = pd.DataFrame(apo)
    apo_table["type"] = "apo"
    apo_table["id"] = np.arange(len(apo_table)) + 1
    apo_table["distance"] = apo_d

    peri_table = pd.DataFrame(peri)
    peri_table["type"] = "peri"
    peri_table["id"] = np.arange(len(peri_table)) + 1
    peri_table["distance"] = peri_d

    items = pd.concat([apo_table, peri_table], axis=0).sort_values("start")

    items = items.reset_index(drop=True)

    intervals = []
    for item_id, per in items.query('type == "peri"').iterrows():
        try:
            before = items.loc[item_id - 1]
            before_start = before["start"]
        except KeyError:
            before_start = per.start - (items.loc[item_id + 1]["start"] - per.start)

        try:
            after = items.loc[item_id + 1]
            after_start = after["start"]
        except KeyError:
            after_start = per.start + (per.start - items.loc[item_id - 1]["start"])


        item = per.to_dict()
        item.update(
            {
                "start": before_start,
                "end": after_start,
                # "type": "peri",
                # "id": per["id"],
                # "body": target,
                "ca": per.start,
            },
        )
        intervals.append(item)

    intervals = pd.DataFrame(intervals)

    return intervals, peri_table, apo_table


def find_ca(start, end, target="JUPITER", observer="JUICE_JANUS") -> pd.Timestamp:
    """Returns the closest approaches within start and end."""
    dist = Distance(observer, target)

    c = MinMaxConstraint(dist, MinMaxConditionTypes.GLOBAL_MINIMUM)
    w = SpiceWindow.from_start_end(start, end)

    ca = c.solve(w)
    if not ca.is_point():
        msg = "Could not determine CA."
        raise ValueError(msg)

    return ca.to_start_end()[0]


def juice_flybys_table() -> pd.DataFrame:
    """Find and return a table with the flybys of JUICE.

    These datetimes are not really special ones...
    """
    allwin = []

    for b in ["callisto", "ganymede", "europa"]:
        cas = flybys_cas(
            b,
            start="2030-01-01T00:00:00.000",
            end="2034-10-30T22:30:44.625",
            altitude_treshold_km=80000,
        )
        windows = flybys_windows(
            b,
            start="2030-01-01T00:00:00.000",
            end="2034-10-30T22:30:44.625",
            altitude_treshold_km=80000,
        )
        for ca, w in zip(cas, windows, strict=True):
            # print(w)
            start, end = w.to_start_end()
            allwin.append({"body": b, "start": start, "end": end, "ca": ca})

        log.info(f"{b}: {len(cas)}")

    wins = pd.DataFrame(allwin)

    wins["flyby_id"] = wins.groupby("body").cumcount() + 1

    return wins
