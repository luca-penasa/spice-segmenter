from typing import Iterable

import spiceypy
from attr import define, field
from datetimerange import DateTimeRange
from spiceypy import Cell_Double

from .types import times_types
from .utils import et


class SpiceWindowIter:
    """Iterator for SpiceWindow"""

    def __init__(self, spice_window: "SpiceWindow"):
        self._window = spice_window
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
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

    spice_window: Cell_Double = field(default=None)
    size: int = field(default=2000)  # 1000 intervals
    _default_size: int = field(default=2000, init=False)

    def __attrs_post_init__(self) -> None:
        if not self.spice_window:
            if not self.size:
                self.size = self._default_size
            self.spice_window = Cell_Double(self.size)

    @classmethod
    def from_datetimerange(cls, ranges: list[DateTimeRange]):
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

    def __repr__(self) -> str:
        return f"SpiceWindow({self.start} to {self.end}, N: {len(self)})"

    def __iter__(self) -> Iterable:
        return SpiceWindowIter(self)

    def __deepcopy__(self, memo) -> "SpiceWindow":
        cls = self.__class__
        newobj = cls.__new__(cls)
        newobj.spice_window = spiceypy.copy(self.spice_window)
        memo[id(self)] = newobj
        return newobj

    def __copy__(self) -> "SpiceWindow":
        return SpiceWindow(self.spice_window)

    def __add__(self, other) -> "SpiceWindow":
        return self.union(other)

    def add_interval(self, start: times_types, end: times_types):
        spiceypy.wninsd(et(start), et(end), self.spice_window)

    def intersect(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wnintd(self.spice_window, other.spice_window))

    def union(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wnunid(self.spice_window, other.spice_window))

    def difference(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wndifd(self.spice_window, other.spice_window))

    def compare(self, other: "SpiceWindow", operator: str) -> bool:
        return spiceypy.wnreld(self.spice_window, operator, other.spice_window)

    def complement(self, other=None) -> "SpiceWindow":
        if other is None:
            other = self

        start = other.start
        end = other.end

        return SpiceWindow(spiceypy.wncomd(start, end, self.spice_window))

    def includes(self, start: times_types, end: times_types) -> "SpiceWindow":
        return spiceypy.wnincd(et(start), et(end), self.spice_window)

    def remove_small_intervals(self, min_size: float) -> None:
        spiceypy.wnfltd(min_size, self.spice_window)

    def fill_small_gaps(self, min_size: float) -> None:
        spiceypy.wnfild(min_size, self.spice_window)

    def __getitem__(self, item: int) -> "SpiceWindow":
        if item >= len(self):
            raise IndexError(f"index {item} out of range")
        left, right = spiceypy.wnfetd(self.spice_window, item)
        out = SpiceWindow(size=2)
        out.add_interval(left, right)
        return out

    def to_datetimerange(self) -> list[DateTimeRange]:
        return [
            DateTimeRange(spiceypy.et2datetime(i.start), spiceypy.et2datetime(i.end))
            for i in self
        ]

    @property
    def end(self):
        if len(self) == 0:
            return None

        return self.spice_window[-1]

    @property
    def start(self):
        if len(self) == 0:
            return None

        return self.spice_window[0]

    def contains(self, point: times_types):
        return spiceypy.wnelmd(et(point), self.spice_window)

    def __len__(self):
        return spiceypy.wncard(self.spice_window)

    def plot(self, ax=None, **kwargs) -> list:
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
            plotted.append(plt.axvspan(s, e, **kwargs))

        return plotted
