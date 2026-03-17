"""Tests for TimeSegmentsCollection (public API) and the internal SpiceWindow bridge."""

import copy
import os
import tempfile
from pathlib import Path

import pytest
import spiceypy
from datetimerange import DateTimeRange
from pandas import Timestamp

from spice_segmenter.core.time_segments_collection import TimeSegmentsCollection
from spice_segmenter.core.time_segment import TimeSegment
# SpiceWindow is internal — imported here solely to test the SPICE cell bridge
from spice_segmenter.core.spice_window import SpiceWindow

from . import tour_config as tc


# ============================================================
# TimeSegment
# ============================================================

class TestTimeInterval:
    def test_basic_construction(self):
        iv = TimeSegment("2032-01-01", "2032-06-30")
        assert iv.start == Timestamp("2032-01-01")
        assert iv.end == Timestamp("2032-06-30")

    def test_from_start_end(self):
        iv = TimeSegment.from_start_end("2032-01-01", "2032-06-30")
        assert iv.start == Timestamp("2032-01-01")

    def test_from_et_roundtrip(self):
        iv = TimeSegment("2032-01-01", "2032-06-30")
        start_et, end_et = iv.to_et()
        iv2 = TimeSegment.from_et(start_et, end_et)
        assert abs((iv2.start - iv.start).total_seconds()) < 1
        assert abs((iv2.end - iv.end).total_seconds()) < 1

    def test_duration(self):
        iv = TimeSegment("2032-01-01", "2032-01-02")
        assert iv.duration is not None
        assert iv.duration.days == 1

    def test_contains(self):
        iv = TimeSegment("2032-01-01", "2032-01-10")
        assert iv.contains("2032-01-05")
        assert not iv.contains("2032-02-01")

    def test_frozen(self):
        from attrs.exceptions import FrozenInstanceError
        iv = TimeSegment("2032-01-01", "2032-06-30")
        with pytest.raises(FrozenInstanceError):
            iv.start = Timestamp("2033-01-01")  # type: ignore[misc]

    def test_repr(self):
        iv = TimeSegment("2032-01-01", "2032-01-02")
        r = repr(iv)
        assert "TimeSegment" in r
        assert "→" in r


# ============================================================
# TimeSegmentsCollection — construction
# ============================================================

class TestEventWindowConstruction:
    def test_empty(self):
        w = TimeSegmentsCollection()
        assert len(w) == 0
        assert w.is_empty

    def test_from_start_end(self):
        w = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-30")
        assert len(w) == 1
        assert w[0].start == Timestamp("2032-01-01")
        assert w[0].end == Timestamp("2032-06-30")

    def test_from_intervals(self):
        iv1 = TimeSegment("2032-01-01", "2032-03-01")
        iv2 = TimeSegment("2032-06-01", "2032-09-01")
        w = TimeSegmentsCollection.from_intervals([iv1, iv2])
        assert len(w) == 2
        assert w[0].start == iv1.start
        assert w[1].end == iv2.end

    def test_from_datetimerange(self):
        r = DateTimeRange(Timestamp("2032-01-01"), Timestamp("2032-06-30"))
        w = TimeSegmentsCollection.from_datetimerange([r])
        assert len(w) == 1
        assert w[0].start == Timestamp("2032-01-01")

    def test_from_datetimerange_rejects_none(self):
        with pytest.raises(ValueError):
            TimeSegmentsCollection.from_datetimerange([DateTimeRange(None, None)])


# ============================================================
# TimeSegmentsCollection — collection protocol
# ============================================================

class TestEventWindowCollection:
    def setup_method(self, _):
        self.w = (
            TimeSegmentsCollection.from_start_end("2032-01-01", "2032-03-01")
            + TimeSegmentsCollection.from_start_end("2032-06-01", "2032-09-01")
        )

    def test_len(self):
        assert len(self.w) == 2

    def test_iter_yields_time_segments(self):
        for iv in self.w:
            assert isinstance(iv, TimeSegment)
            assert isinstance(iv.start, Timestamp)
            assert isinstance(iv.end, Timestamp)

    def test_getitem(self):
        assert self.w[0].start == Timestamp("2032-01-01")
        assert self.w[1].end == Timestamp("2032-09-01")

    def test_mixin_total_duration(self):
        td = self.w.total_duration
        assert td.days > 0

    def test_mixin_start_end(self):
        assert self.w.start == Timestamp("2032-01-01")
        assert self.w.end == Timestamp("2032-09-01")


# ============================================================
# TimeSegmentsCollection — set operations
# ============================================================

class TestEventWindowSetOps:
    def test_union_disjoint(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-03-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-06-01", "2032-09-01")
        w3 = w1 + w2
        assert len(w3) == 2

    def test_union_overlapping_merges(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-03-01", "2032-09-01")
        union = w1 + w2
        assert len(union) == 1

    def test_intersect(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-03-01", "2032-09-01")
        ix = w1.intersect(w2)
        assert len(ix) == 1
        assert ix[0].start == Timestamp("2032-03-01")
        assert ix[0].end == Timestamp("2032-06-01")

    def test_difference(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-03-01", "2032-09-01")
        diff = w1.difference(w2)
        assert len(diff) == 1
        assert diff[0].start == Timestamp("2032-01-01")
        assert diff[0].end == Timestamp("2032-03-01")

    def test_complement(self):
        bounds = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-12-31")
        w = TimeSegmentsCollection.from_start_end("2032-06-01", "2032-09-01")
        comp = w.complement(bounds)
        assert len(comp) == 2

    def test_compare_equal(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        assert w1 == w2
        assert w1.compare(w2, "=")

    def test_compare_not_equal(self):
        w1 = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = TimeSegmentsCollection.from_start_end("2032-07-01", "2032-09-01")
        assert w1 != w2
        assert w1.compare(w2, "<>")


# ============================================================
# TimeSegmentsCollection — mutating filters
# ============================================================

class TestEventWindowFilters:
    def test_remove_small_intervals(self):
        normal = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-01-05")
        tiny = TimeSegmentsCollection.from_start_end("2032-02-01 00:00:00", "2032-02-01 00:00:05")
        combined = normal + tiny
        assert len(combined) == 2
        combined.remove_small_intervals(10.0)   # 10 s threshold removes the tiny one
        assert len(combined) == 1

    def test_fill_small_gaps(self):
        w = (
            TimeSegmentsCollection.from_start_end("2032-01-01", "2032-01-05")
            + TimeSegmentsCollection.from_start_end("2032-01-05 00:00:01", "2032-01-10")
        )
        assert len(w) == 2
        w.fill_small_gaps(2.0)      # fill gaps < 2 s
        assert len(w) == 1


# ============================================================
# TimeSegmentsCollection — point queries
# ============================================================

class TestEventWindowQuery:
    def setup_method(self, _):
        self.w = (
            TimeSegmentsCollection.from_start_end("2032-01-01", "2032-03-01")
            + TimeSegmentsCollection.from_start_end("2032-06-01", "2032-09-01")
        )

    def test_contains_inside(self):
        assert self.w.contains("2032-02-01")
        assert self.w.contains("2032-07-15")

    def test_contains_outside(self):
        assert not self.w.contains("2032-04-15")   # gap
        assert not self.w.contains("2033-01-01")

    def test_call_mask(self):
        import numpy as np
        import pandas as pd

        times = [
            pd.Timestamp("2032-02-01"),
            pd.Timestamp("2032-04-15"),
            pd.Timestamp("2032-07-15"),
        ]
        mask = self.w(np.array(times))
        assert mask[0]
        assert not mask[1]
        assert mask[2]

    def test_includes(self):
        assert self.w.includes("2032-01-15", "2032-02-15")
        assert not self.w.includes("2032-01-15", "2032-04-01")  # crosses gap


# ============================================================
# TimeSegmentsCollection — export
# ============================================================

class TestEventWindowExport:
    def setup_method(self, _):
        self.w = (
            TimeSegmentsCollection.from_start_end("2022-12-01T12:22:12", "2022-12-02")
            + TimeSegmentsCollection.from_start_end("2023-12-01", "2023-12-02T12:02:11")
        )

    def test_to_pandas_shape(self):
        df = self.w.to_pandas(round_to="s")
        assert list(df.columns) == ["start", "end"]
        assert len(df) == 2
        assert df.iloc[0].start.year == 2022
        assert df.iloc[1].end.year == 2023

    def test_to_pandas_empty(self):
        df = TimeSegmentsCollection().to_pandas()
        assert len(df) == 0
        assert list(df.columns) == ["start", "end"]

    def test_to_datetimerange(self):
        ranges = self.w.to_datetimerange()
        assert len(ranges) == 2
        assert isinstance(ranges[0], DateTimeRange)
        assert ranges[0].start_datetime.year == 2022  # type: ignore[union-attr]
        assert ranges[1].end_datetime.year == 2023     # type: ignore[union-attr]

    def test_datetimerange_roundtrip(self):
        ranges = self.w.to_datetimerange()
        w2 = TimeSegmentsCollection.from_datetimerange(ranges)
        assert self.w == w2

    def test_to_juice_core_csv(self):
        w = TimeSegmentsCollection.from_start_end(
            "2023-11-02T12:00:00.000", "2023-11-02T12:00:01.002"
        )
        tmpfile = tempfile.gettempdir() + "/test_event.csv"
        if Path(tmpfile).exists():
            os.remove(tmpfile)
        w.to_juice_core_csv(tmpfile)
        assert Path(tmpfile).exists()

    def test_iter_matches_pandas(self):
        df = self.w.to_pandas(round_to="ms")
        for i, iv in enumerate(self.w):
            assert iv.start.round("ms") == df.iloc[i]["start"]
            assert iv.end.round("ms") == df.iloc[i]["end"]


# ============================================================
# TimeSegmentsCollection — copy semantics
# ============================================================

class TestEventWindowCopy:
    def test_shallow_copy(self):
        w = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = copy.copy(w)
        assert w == w2
        assert w is not w2

    def test_deepcopy_independent(self):
        w = TimeSegmentsCollection.from_start_end("2032-01-01", "2032-06-01")
        w2 = copy.deepcopy(w)
        assert w == w2
        w2._segments_.clear()
        assert len(w) == 1


# ============================================================
# TimeSegmentsCollection — plot (smoke)
# ============================================================

def test_time_segments_collection_plot():
    w = (
        TimeSegmentsCollection.from_start_end("2032-01-01", "2032-03-01")
        + TimeSegmentsCollection.from_start_end("2032-06-01", "2032-09-01")
    )
    patches = w.plot()
    assert len(patches) == len(w)
    w.plot(label="test_label")


# ============================================================
# TimeSegmentsCollection — tour_config integration (requires SPICE kernels)
# ============================================================

def test_time_segments_collection_from_tour_coverage():
    s, e = tc.coverage
    w = TimeSegmentsCollection.from_start_end(s, e)
    assert len(w) == 1
    assert w.start is not None
    assert w.end is not None


# ============================================================
# Internal SpiceWindow (SPICE-cell semantics)
# ============================================================

class TestSpiceWindowInternal:
    """SpiceWindow is internal.  These tests exercise the SPICE cell bridge
    that TimeSegmentsCollection delegates to for set operations."""

    def test_basic_ops(self):
        w1 = SpiceWindow()
        w1.add_interval(0, 10)
        w2 = SpiceWindow()
        w2.add_interval(20, 30)
        w3 = w1 + w2
        assert w3.compare(w1, operator=">")
        assert w1.compare(w2, operator="<>")
        assert len(w3) == 2
        assert len(w1) == 1

    def test_size_overflow(self):
        with pytest.raises(spiceypy.utils.exceptions.SpiceWINDOWEXCESS):
            w = SpiceWindow(size=1)
            w.add_interval(0, 10)

    def test_default_size(self):
        w = SpiceWindow()
        assert w.size == w._default_size

    def test_raw_cell_constructor(self):
        cell = spiceypy.stypes.Cell_Double(12)
        w = SpiceWindow(cell)
        assert w.spice_window == cell

    def test_repr(self):
        w = SpiceWindow()
        s = "2023-11-02T12:00:00.000"
        e = "2023-11-02T12:00:01.002"
        w.add_interval(s, e)
        assert w.__repr__() == f"SpiceWindow({s} to {e}, N: 1)"

    def test_copy(self):
        w = SpiceWindow()
        w.add_interval(0, 10)
        w_copy = copy.copy(w)
        w_deep = copy.deepcopy(w)
        assert id(w.spice_window) == id(w_copy.spice_window)
        assert id(w.spice_window) != id(w_deep.spice_window)
        assert w == w_copy
        assert w == w_deep
        w_deep.add_interval(20, 30)
        assert w.spice_window != w_deep.spice_window

    def test_complement(self):
        w = SpiceWindow()
        w.add_interval(0, 10)
        w.add_interval(20, 30)
        expected = SpiceWindow()
        expected.add_interval(10, 20)
        assert expected == w.complement(w)

    def test_contains(self):
        w = SpiceWindow()
        for s, e in [(0, 10), (11, 12), (14, 16)]:
            w.add_interval(s, e)
        assert w.contains(2)
        assert not w.contains(22)
        assert not w.contains(10.5)

    def test_filters(self):
        w = SpiceWindow()
        w.add_interval(0, 10)
        w.add_interval(11, 30)
        w.fill_small_gaps(2)
        assert len(w) == 1
        w.add_interval(31, 31.1)
        w.remove_small_intervals(0.2)
        assert len(w) == 1

    def test_from_datetimerange(self):
        from planetary_coverage import et
        s, e = tc.coverage
        trange = DateTimeRange(Timestamp(s), Timestamp(e))
        w = SpiceWindow.from_datetimerange([trange])
        assert w.start == et(tc.coverage[0])
        assert w.end == et(tc.coverage[1])
        assert len(w) == 1
        with pytest.raises(ValueError):
            SpiceWindow.from_datetimerange([DateTimeRange(None, None)])
