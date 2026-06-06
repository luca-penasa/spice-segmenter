"""Mutable collection of time intervals — the public API for event windows."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from attr import define, field
from datetimerange import DateTimeRange
from time_segments import SegmentsCollectionMixin

from ..support.time_types import TIMES_TYPES
from .time_segment import TimeSegment


@define(repr=False, order=False, eq=False)
class TimeSegmentsCollection(SegmentsCollectionMixin[TimeSegment]):
    """A mutable, ordered collection of non-overlapping time intervals.

    This is the public replacement for ``SpiceWindow``, serving as both the
    *search window* passed into ``Constraint.solve()`` and the *result* it
    returns.

    Internally, SPICE set operations (union, intersection, complement, etc.)
    are delegated to an ephemeral :class:`~spice_segmenter.core.SpiceWindow`
    cell.  No SPICE state is retained between calls.

    The class also inherits the full feature set of
    :class:`~time_segments.SegmentsCollectionMixin`, including:

    * ``total_duration``, ``start`` / ``end``, ``timespan``
    * ``gaps()``, ``filter()``, ``filter_by_time_range()``
    * ``merge_overlapping()``, ``split_at_time()``
    * ``find_overlaps()``, ``has_overlaps()``

    Construction
    ------------
    >>> TimeSegmentsCollection.from_start_end("2032-01-01", "2033-01-01")    # single-interval search window
    >>> TimeSegmentsCollection.from_intervals([iv1, iv2, iv3])
    >>> TimeSegmentsCollection.from_datetimerange(list_of_datetimeranges)

    Iteration yields :class:`TimeSegment` objects with ``start`` / ``end``
    as ``pd.Timestamp``:

    >>> for iv in window:
    ...     print(iv.start, iv.end, iv.duration)
    """

    _segments_: list[TimeSegment] = field(factory=list, alias="segments")

    # ------------------------------------------------------------------
    # Internal SPICE bridge (not part of the public API)
    # ------------------------------------------------------------------

    def _to_spice_window(self) -> SpiceWindow:  # type: ignore[name-defined]  # noqa: F821
        """Materialise a :class:`SpiceWindow` from the current segments."""
        from .spice_window import SpiceWindow

        sw = SpiceWindow()
        for iv in self._segments_:
            sw.add_interval(iv.start, iv.end)
        return sw

    @classmethod
    def _from_spice_window(cls, sw: SpiceWindow) -> TimeSegmentsCollection:  # type: ignore[name-defined]  # noqa: F821
        """Create an ``TimeSegmentsCollection`` from a :class:`SpiceWindow`."""
        intervals = [
            TimeSegment.from_et(sw[i].start, sw[i].end) for i in range(len(sw))
        ]
        return cls(segments=intervals)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_start_end(cls, start: TIMES_TYPES, end: TIMES_TYPES) -> TimeSegmentsCollection:
        """Build a single-interval ``TimeSegmentsCollection`` — the typical search window.

        Parameters
        ----------
        start, end:
            Any time type accepted by ``pd.Timestamp``, a SPICE ET float, or an
            ISO-8601 string.
        """
        return cls(segments=[TimeSegment(start=start, end=end)])  # type: ignore[arg-type]

    @classmethod
    def from_intervals(cls, intervals: list[TimeSegment]) -> TimeSegmentsCollection:
        """Build an ``TimeSegmentsCollection`` from a list of :class:`TimeSegment` objects."""
        return cls(segments=list(intervals))

    @classmethod
    def from_datetimerange(cls, ranges: list[DateTimeRange]) -> TimeSegmentsCollection:
        """Build an ``TimeSegmentsCollection`` from a list of ``DateTimeRange`` objects."""
        intervals = []
        for r in ranges:
            if r.start_datetime is None or r.end_datetime is None:
                raise ValueError(
                    "DateTimeRange must have valid start and end datetimes",
                )
            intervals.append(
                TimeSegment(
                    start=pd.Timestamp(r.start_datetime),
                    end=pd.Timestamp(r.end_datetime),
                ),
            )
        return cls(segments=intervals)

    # ------------------------------------------------------------------
    # Required by SegmentsCollectionMixin
    # ------------------------------------------------------------------

    def _create_empty_segment(self, start: Any, end: Any) -> TimeSegment:
        return TimeSegment(start=start, end=end)  # type: ignore[arg-type]

    def _create_new_collection(self, segments: list[TimeSegment]) -> TimeSegmentsCollection:
        return TimeSegmentsCollection(segments=segments)

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._segments_)

    def __iter__(self) -> Iterator[TimeSegment]:
        return iter(self._segments_)

    def __getitem__(self, item: int) -> TimeSegment:
        return self._segments_[item]

    # ------------------------------------------------------------------
    # Set operations (delegated to the internal SpiceWindow bridge)
    # ------------------------------------------------------------------

    def union(self, other: TimeSegmentsCollection) -> TimeSegmentsCollection:
        """Return the union of this window and *other*."""
        return TimeSegmentsCollection._from_spice_window(
            self._to_spice_window().union(other._to_spice_window()),
        )

    def intersect(self, other: TimeSegmentsCollection) -> TimeSegmentsCollection:
        """Return the intersection of this window and *other*."""
        return TimeSegmentsCollection._from_spice_window(
            self._to_spice_window().intersect(other._to_spice_window()),
        )

    def difference(self, other: TimeSegmentsCollection) -> TimeSegmentsCollection:
        """Return this window minus *other*."""
        return TimeSegmentsCollection._from_spice_window(
            self._to_spice_window().difference(other._to_spice_window()),
        )

    def complement(self, bounds: TimeSegmentsCollection | None = None) -> TimeSegmentsCollection:
        """Return the complement of this window within *bounds*.

        If *bounds* is ``None`` the complement is taken with respect to this
        window's own encompassing interval.
        """
        bounds_sw = (
            bounds._to_spice_window()
            if bounds is not None
            else self._to_spice_window()
        )
        return TimeSegmentsCollection._from_spice_window(
            self._to_spice_window().complement(bounds_sw),
        )

    def compare(self, other: TimeSegmentsCollection, operator: str) -> bool:
        """Compare two windows using SPICE ``wnreld`` operators.

        Operators: ``"="``, ``"<>"``, ``"<"``, ``">"``, ``"<="``, ``">="``
        """
        return self._to_spice_window().compare(other._to_spice_window(), operator)

    def __add__(self, other: TimeSegmentsCollection) -> TimeSegmentsCollection:
        return self.union(other)

    # ------------------------------------------------------------------
    # In-place mutating filters
    # ------------------------------------------------------------------

    def remove_small_intervals(self, min_size_s: float) -> None:
        """Remove all intervals shorter than *min_size_s* seconds in-place."""
        sw = self._to_spice_window()
        sw.remove_small_intervals(min_size_s)
        self._segments_ = TimeSegmentsCollection._from_spice_window(sw)._segments_

    def fill_small_gaps(self, min_size_s: float) -> None:
        """Fill all gaps shorter than *min_size_s* seconds in-place."""
        sw = self._to_spice_window()
        sw.fill_small_gaps(min_size_s)
        self._segments_ = TimeSegmentsCollection._from_spice_window(sw)._segments_

    # ------------------------------------------------------------------
    # Point / interval queries
    # ------------------------------------------------------------------

    def contains(self, point: TIMES_TYPES) -> bool:
        """Return ``True`` if *point* falls within any interval."""
        pt = pd.Timestamp(str(point))  # type: ignore[arg-type]
        return any(seg.contains(pt) for seg in self._segments_)

    def includes(self, start: TIMES_TYPES, end: TIMES_TYPES) -> bool:
        """Return ``True`` if the interval [*start*, *end*] is fully covered."""
        return self._to_spice_window().includes(start, end)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask for an array of time points."""
        points = np.atleast_1d(points)
        return np.array([self.contains(p) for p in points])

    @property
    def point_events(self) -> list[TimeSegment]:
        """All zero-duration segments in this collection (e.g. closest approaches, extrema).

        A segment is considered a point event when its duration is shorter
        than 1 µs — the threshold used by :attr:`TimeSegment.is_point`.

        Examples
        --------
        >>> result = (Distance(...) == MinMaxConditionTypes.GLOBAL_MINIMUM).solve(window)
        >>> result.point_events   # list of TimeSegment with is_point == True
        """
        return [seg for seg in self._segments_ if seg.is_point]

    @property
    def intervals(self) -> list[TimeSegment]:
        """All non-zero-duration segments (regular time intervals, not point events)."""
        return [seg for seg in self._segments_ if not seg.is_point]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_datetimerange(self) -> list[DateTimeRange]:
        """Convert to a list of ``DateTimeRange`` objects."""
        return [DateTimeRange(seg.start, seg.end) for seg in self._segments_]

    def to_pandas(self, round_to: str | None = "s") -> pd.DataFrame:
        """Export as a DataFrame with ``start`` and ``end`` ``pd.Timestamp`` columns.

        If any segments carry ``label``, ``value``, or ``property_name``
        metadata those columns are included automatically.

        Parameters
        ----------
        round_to:
            Pandas frequency string for rounding (``"s"`` for seconds,
            ``"ms"`` for milliseconds, etc.).  Pass ``None`` to suppress
            rounding.
        """
        if len(self) == 0:
            return pd.DataFrame(columns=["start", "end"])

        has_meta = any(
            seg.label or seg.value is not None or seg.property_name
            for seg in self._segments_
        )
        rows = []
        for seg in self._segments_:
            row: dict = {"start": seg.start, "end": seg.end}
            if has_meta:
                row["label"] = seg.label
                row["value"] = seg.value
                row["property_name"] = seg.property_name
            rows.append(row)
        tab = pd.DataFrame(rows)
        if round_to:
            tab["start"] = tab["start"].dt.round(round_to)
            tab["end"] = tab["end"].dt.round(round_to)
        return tab

    def to_juice_core_csv(
        self,
        filename: str,
        obs_id: str = "OBSERVATION",
        wg: str = "WG2",
        add_z: bool = True,
    ) -> None:
        """Write intervals to a JUICE core CSV file."""
        t = self.to_pandas()
        t["id"] = obs_id
        t["unk"] = ""
        t["wg"] = wg
        t = t[["id", "start", "end", "unk", "wg"]]
        dformat = "%Y-%m-%dT%H:%M:%S"
        if add_z:
            dformat += "Z"
        t.to_csv(filename, date_format=dformat, header=False, index=False)

    def plot(
        self, ax: matplotlib.axes.Axes | None = None, **kwargs: Any,
    ) -> list:
        """Plot each interval as an ``axvspan`` patch.

        The first interval uses the provided ``label`` (if any); subsequent
        intervals get a leading underscore so they do not duplicate the legend
        entry.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        plotted = []
        for i, seg in enumerate(self._segments_):
            kw = dict(kwargs)
            if "label" in kw and i > 0:
                kw["label"] = f"_{kw['label']}"
            plotted.append(plt.axvspan(seg.start, seg.end, **kw))
        return plotted

    # ------------------------------------------------------------------
    # Dunder / magic methods
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSegmentsCollection):
            return False
        return self.compare(other, "=")

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "TimeSegmentsCollection(empty)"
        return f"TimeSegmentsCollection(N={n}, {self.start} → {self.end}, total={self.total_duration})"

    def __copy__(self) -> TimeSegmentsCollection:
        return TimeSegmentsCollection(segments=list(self._segments_))

    def __deepcopy__(self, memo: dict) -> TimeSegmentsCollection:
        new_segs = [copy.deepcopy(seg, memo) for seg in self._segments_]
        obj = TimeSegmentsCollection(segments=new_segs)
        memo[id(self)] = obj
        return obj
