import spiceypy
from attr import define, field
from datetimerange import DateTimeRange
from planetary_coverage import et
from spiceypy import Cell_Double

from .types import times_types


class SpiceWindowIter:
    """Iterator for SpiceWindow"""

    def __init__(self, spice_window):
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


# @define
# class SpiceInterval:
#     _start: float = field(default=None, converter=et)
#     _end: float = field(default=None, converter=et)
#
#     _data: Cell_Double = field(factory=partial(Cell_Double, 2))
#
#     def __attrs_post_init__(self):
#         if self._start is not None and self._end is not None:
#             self._data[0] = self._start
#             self._data[1] = self._end
#         elif self._data is not None:
#             self._start = self._data[0]
#             self._end = self._data[1]
#         else:
#             raise ValueError("Either start and end or data must be provided")
#
#     @property
#     def start(self):
#         return self._data[0]
#
#     @property
#     def end(self):
#         return self._data[1]


@define(repr=False, order=False, eq=False)
class SpiceWindow:
    """Represents a collection of intervals in which some conditions apply."""

    _spice_window: Cell_Double = field(default=None)
    _size: int = field(default=2000)  # 1000 intervals
    _default_size: int = field(default=2000, init=False)

    @property
    def spice_window(self):
        return self._spice_window

    def __attrs_post_init__(self):
        if not self._spice_window:
            if not self._size:
                self._size = self._default_size
            self._spice_window = Cell_Double(self._size)

    @classmethod
    def from_datetimerange(cls, ranges=list[DateTimeRange]):
        """Create a SpiceWindow from a list of DateTimeRanges"""
        window = cls()
        for r in ranges:
            window.add_interval(
                r.start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                r.end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        return window

    def __repr__(self):
        return f"SpiceWindow({self.start} to {self.end}, N: {len(self)})"

    def __iter__(self):
        return SpiceWindowIter(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        newobj = cls.__new__(cls)
        newobj._spice_window = spiceypy.copy(self._spice_window)
        memo[id(self)] = newobj
        return newobj

    def __copy__(self):
        return SpiceWindow(self._spice_window)

    def __add__(self, other):
        return self.union(other)

    def add_interval(self, start: times_types, end: times_types):
        spiceypy.wninsd(et(start), et(end), self._spice_window)

    def intersect(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wnintd(self._spice_window, other._spice_window))

    def union(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wnunid(self._spice_window, other._spice_window))

    def difference(self, other: "SpiceWindow") -> "SpiceWindow":
        return SpiceWindow(spiceypy.wndifd(self._spice_window, other._spice_window))

    def compare(self, other: "SpiceWindow", operator: str) -> bool:
        return spiceypy.wnreld(self._spice_window, operator, other._spice_window)

    def complement(self, other) -> "SpiceWindow":
        return SpiceWindow(spiceypy.wncomd(self._spice_window, other._spice_window))

    def includes(self, start: times_types, end: times_types) -> "SpiceWindow":
        return spiceypy.wnincd(et(start), et(end), self._spice_window)

    def remove_small_intervals(self, min_size: float) -> None:
        spiceypy.wnfltd(min_size, self._spice_window)

    def fill_small_gaps(self, min_size: float) -> None:
        spiceypy.wnfild(min_size, self._spice_window)

    def __getitem__(self, item: int) -> "SpiceWindow":
        if item >= len(self):
            raise IndexError(f"index {item} out of range")
        left, right = spiceypy.wnfetd(self._spice_window, item)
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

        # newin = deepcopy(self)
        # spiceypy.wnextd("R", newin._spice_window)
        # return newin._spice_window[-1]

        return self._spice_window[-1]

    @property
    def start(self):
        if len(self) == 0:
            return None

        # newin = deepcopy(self)
        # got = spiceypy.wnextd("L", newin._spice_window)
        # return newin._spice_window[0]

        return self._spice_window[0]

    def contains(self, point: times_types):
        return spiceypy.wnelmd(et(point), self._spice_window)

    def __len__(self):
        return spiceypy.wncard(self._spice_window)
