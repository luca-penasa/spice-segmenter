"""Atomic, immutable time interval with pandas-native timestamps."""

from __future__ import annotations

from typing import Any

import pandas as pd
from attr import define, field
from planetary_coverage import utc
from time_segments import TimeSegmentMixin

from ..support.spice_utilities import et
from ..support.time_types import TIMES_TYPES

_POINT_THRESHOLD_S = 1e-6  # 1 µs — any segment shorter than this is a point event


def _to_timestamp(value: object) -> pd.Timestamp:
    """Convert any supported time representation to ``pd.Timestamp``."""
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, float)):
        # Assume SPICE ephemeris time (ET)
        return pd.Timestamp(utc(value))
    return pd.Timestamp(str(value))  # type: ignore[arg-type]


@define(frozen=True, repr=False, order=False)
class TimeSegment(TimeSegmentMixin):
    """An atomic, immutable time interval with pandas-native timestamps.

    Extends :class:`~time_segments.TimeSegmentMixin`, which provides a rich
    set of read-only operations: ``duration``, ``middle``, ``contains``,
    ``intersects``, ``intersect``, etc.

    .. note::
       The interval is **frozen**: mutating methods inherited from the mixin
       (``set_duration``, ``set_times_from_midpoint``, …) raise
       ``attrs.exceptions.FrozenInstanceError``.  Use :class:`TimeSegmentsCollection`
       to build mutable collections.

    A ``TimeSegment`` whose ``start == end`` (within 1 µs) is called a
    *point event* — for example a closest approach, a local extremum of a
    property, or a FOV ingress/egress.  Use :attr:`is_point` to test for this
    and :meth:`at_time` to construct one conveniently.

    Parameters
    ----------
    start, end:
        Any time type accepted by ``pd.Timestamp``, a SPICE ET float, or an
        ISO-8601 string.  Stored internally as ``pd.Timestamp``.
    label:
        Short human-readable description, e.g. ``"closest approach"``.
    value:
        Property value evaluated at this instant (e.g. distance in km at CA).
    property_name:
        Name of the property that produced this segment/event.
    metadata:
        Free-form extra attributes.

    Examples
    --------
    >>> seg = TimeSegment("2032-01-01", "2032-06-30")
    >>> seg.duration
    Timedelta('181 days 00:00:00')

    >>> ca = TimeSegment.at_time("2032-01-15T04:23:11", label="closest approach", value=1234.5, property_name="distance")
    >>> ca.is_point
    True
    """

    start: pd.Timestamp = field(converter=_to_timestamp)
    end: pd.Timestamp = field(converter=_to_timestamp)
    label: str = field(default="", kw_only=True)
    value: float | None = field(default=None, kw_only=True)
    property_name: str = field(default="", kw_only=True)
    metadata: dict[str, Any] = field(factory=dict, kw_only=True)

    # ------------------------------------------------------------------
    # Point-event helpers
    # ------------------------------------------------------------------

    @property
    def is_point(self) -> bool:
        """``True`` if this segment represents a single instant (duration < 1 µs)."""
        return self.duration.total_seconds() < _POINT_THRESHOLD_S

    @classmethod
    def at_time(
        cls,
        time: TIMES_TYPES,
        *,
        label: str = "",
        value: float | None = None,
        property_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TimeSegment:
        """Create a zero-duration (point) ``TimeSegment`` at a single instant.

        Parameters
        ----------
        time:
            The instant, as any supported time type.
        label:
            Human-readable description (e.g. ``"closest approach"``).
        value:
            Property value at this instant.
        property_name:
            Name of the property that produced this event.
        metadata:
            Extra attributes.
        """
        t = _to_timestamp(time)
        return cls(
            start=t,
            end=t,
            label=label,
            value=value,
            property_name=property_name,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_et(cls, start_et: float, end_et: float) -> TimeSegment:
        """Create a ``TimeSegment`` from SPICE ephemeris time (ET) values."""
        return cls(
            start=pd.Timestamp(utc(start_et)),
            end=pd.Timestamp(utc(end_et)),
        )

    @classmethod
    def from_start_end(cls, start: TIMES_TYPES, end: TIMES_TYPES) -> TimeSegment:
        """Create a ``TimeSegment`` from any supported time type."""
        return cls(start=start, end=end)  # type: ignore[arg-type]

    def to_et(self) -> tuple[float, float]:
        """Return ``(start_et, end_et)`` as SPICE ephemeris time floats."""
        return et(self.start), et(self.end)

    def __repr__(self) -> str:
        if self.is_point:
            extra = f", label={self.label!r}" if self.label else ""
            extra += f", value={self.value}" if self.value is not None else ""
            return f"TimeSegment(point @ {self.start}{extra})"
        return f"TimeSegment({self.start} → {self.end}, {self.duration})"
