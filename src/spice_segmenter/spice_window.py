from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import spiceypy
from attr import define, field
from datetimerange import DateTimeRange
from planetary_coverage import utc
from spiceypy import Cell_Double, SpiceCell

from .types import times_types
from .utils import et


class SpiceWindowIter:
    """Iterator for SpiceWindow"""

    def __init__(self, spice_window: SpiceWindow):
        self._window = spice_window
        self._index = 0

    def __iter__(self) -> SpiceWindowIter:
        return self

    def __next__(self) -> SpiceWindow:
        if self._index >= len(self._window):
            raise StopIteration
        else:
            win = self._window[self._index]
            self._index += 1
            return win


@define(repr=False, order=False, eq=False)
class SpiceWindow:
    """Represents a collection of intervals in which some conditions apply.

    This is a wrapper around the SPICE time window related routines.
    """

    spice_window: SpiceCell | None = field(default=None)
    size: int = field(default=2000)
    _default_size: int = field(default=2000, init=False)

    def __attrs_post_init__(self) -> None:
        if not self.spice_window:
            if not self.size:
                self.size = self._default_size
            self.spice_window = Cell_Double(self.size)

    @classmethod
    def from_datetimerange(cls, ranges: list[DateTimeRange]) -> SpiceWindow:
        """Create a SpiceWindow from a list of DateTimeRanges"""
        window = cls()
        for r in ranges:
            start = r.start_datetime
            end = r.end_datetime

            if (
                start is None or end is None
            ):  # for some reason, r.start_datetime is marked as Optional[datetime]! better check
                raise ValueError

            window.add_interval(
                start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        return window

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpiceWindow):
            return False

        return self.spice_window == other.spice_window

    def __repr__(self) -> str:
        return f"SpiceWindow({utc(self.start)} to {utc(self.end)}, N: {len(self)})"

    def __iter__(self) -> Iterator[SpiceWindow]:
        return SpiceWindowIter(self)

    def __deepcopy__(self, memo: dict) -> SpiceWindow:
        cls = self.__class__
        newobj = cls.__new__(cls)
        newobj.spice_window = spiceypy.copy(self.spice_window)  # type: ignore
        memo[id(self)] = newobj
        return newobj

    def __copy__(self) -> SpiceWindow:
        return SpiceWindow(self.spice_window)

    def __add__(self, other: SpiceWindow) -> SpiceWindow:
        return self.union(other)

    def add_interval(self, start: times_types, end: times_types) -> None:
        spiceypy.wninsd(et(start), et(end), self.spice_window)

    def intersect(self, other: SpiceWindow) -> SpiceWindow:
        return SpiceWindow(spiceypy.wnintd(self.spice_window, other.spice_window))

    def union(self, other: SpiceWindow) -> SpiceWindow:
        return SpiceWindow(spiceypy.wnunid(self.spice_window, other.spice_window))

    def difference(self, other: SpiceWindow) -> SpiceWindow:
        return SpiceWindow(spiceypy.wndifd(self.spice_window, other.spice_window))

    def compare(self, other: SpiceWindow, operator: str) -> bool:
        return bool(spiceypy.wnreld(self.spice_window, operator, other.spice_window))

    def complement(self, other: SpiceWindow | None = None) -> SpiceWindow:
        if other is None:
            other = self

        start = other.start
        end = other.end

        if (start is None) | (end is None):
            raise ValueError("Cannot compute complement of empty window")

        return SpiceWindow(spiceypy.wncomd(start, end, self.spice_window))

    def includes(self, start: times_types, end: times_types) -> bool:
        return bool(spiceypy.wnincd(et(start), et(end), self.spice_window))

    def remove_small_intervals(self, min_size: float) -> None:
        spiceypy.wnfltd(min_size, self.spice_window)

    def fill_small_gaps(self, min_size: float) -> None:
        spiceypy.wnfild(min_size, self.spice_window)

    def __getitem__(self, item: int) -> SpiceWindow:
        if item >= len(self):
            raise IndexError(f"index {item} out of range")
        left, right = spiceypy.wnfetd(self.spice_window, item)
        out = SpiceWindow(size=2)
        out.add_interval(left, right)
        return out

    def to_datetimerange(self) -> list[DateTimeRange]:
        return [
            DateTimeRange(pd.Timestamp(utc(i.start)), pd.Timestamp(utc(i.end)))
            for i in self
        ]

    @property
    def end(self) -> float | None:
        if len(self) == 0:
            return None

        return float(self.spice_window[-1])

    @property
    def start(self) -> float | None:
        if len(self) == 0:
            return None

        return float(self.spice_window[0])

    def contains(self, point: times_types) -> bool:
        return bool(spiceypy.wnelmd(et(point), self.spice_window))

    def __len__(self) -> int:
        return int(spiceypy.wncard(self.spice_window))

    def plot(self, ax=None, **kwargs) -> list:  # type: ignore
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        intervals = self.to_datetimerange()

        plotted = []
        for i, inter in enumerate(intervals):
            if "label" in kwargs and i == 1:
                kwargs[
                    "label"
                ] = f"_{kwargs['label']}"  # not really nice, we are altering kwargs

            s = inter.start_datetime
            e = inter.end_datetime
            plotted.append(plt.axvspan(s, e, **kwargs))  # type: ignore

        return plotted

    def to_pandas(self, round_to: str = "S") -> pd.DataFrame:
        out = []
        for i in self:
            out.append(
                {"start": np.datetime64(utc(i.start)), "end": np.datetime64(utc(i.end))}
            )

        tab = pd.DataFrame(out)
        if round_to:
            tab.start = tab.start.round(round_to)
            tab.end = tab.end.round(round_to)

        return tab

    def to_juice_core_csv(
        self,
        filename: str,
        obs_id: str = "OBSERVATION",
        wg: str = "WG2",
        add_z: bool = True,
    ) -> None:
        t = self.to_pandas()

        t["id"] = obs_id
        t["unk"] = ""
        t["wg"] = wg

        t = t[["id", "start", "end", "unk", "wg"]]

        dformat = "%Y-%m-%dT%H:%M:%S"
        if add_z:
            dformat += "Z"

        t.to_csv(filename, date_format=dformat, header=False, index=False)
