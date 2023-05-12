import pytest
import spiceypy
from planetary_coverage import utc

from spice_segmenter.spice_window import SpiceWindow

from . import TourConfig

tc = TourConfig()


def test_spice_window() -> None:
    w1 = SpiceWindow()
    w1.add_interval(0, 10)

    w2 = SpiceWindow()
    w2.add_interval(20, 30)

    w3 = w1 + w2
    assert w3.compare(w1, operator=">")
    assert w1.compare(w2, operator="<>")
    assert len(w3) == 2
    assert len(w1) == 1

    with pytest.raises(spiceypy.utils.exceptions.SpiceWINDOWEXCESS):
        w4 = SpiceWindow(size=1)
        w4.add_interval(0, 10)

    ww = SpiceWindow()
    assert ww.size == ww._default_size

    cell = spiceypy.stypes.Cell_Double(12)

    ww = SpiceWindow(cell)
    assert ww.spice_window == cell

    w = SpiceWindow()

    s = "2023-11-02T12:00:00.000"
    e = "2023-11-02T12:00:01.002"
    w.add_interval(s, e)

    text = w.__repr__()
    assert text == f"SpiceWindow({s} to {e}, N: 1)"


def test_copy_window() -> None:
    w = SpiceWindow()

    w.add_interval(0, 10)
    import copy

    w_copy = copy.copy(w)

    w_deepcopy = copy.deepcopy(w)

    assert id(w.spice_window) == id(w_copy.spice_window)
    assert id(w.spice_window) != id(w_deepcopy.spice_window)

    assert id(w) != id(w_copy)
    assert id(w) != id(w_deepcopy)

    assert w.spice_window == w_copy.spice_window
    assert w == w_copy
    assert w.spice_window == w_deepcopy.spice_window
    assert w == w_deepcopy

    w_deepcopy.add_interval(20, 30)

    assert w.spice_window != w_deepcopy.spice_window

    w_copy.add_interval(20, 30)

    assert w.spice_window == w_copy.spice_window


def test_complement_window() -> None:
    w = SpiceWindow()
    w.add_interval(0, 10)
    w.add_interval(20, 30)

    w_complement = SpiceWindow()
    w_complement.add_interval(10, 20)

    assert w_complement == w.complement(w)


def test_contains_window() -> None:
    w = SpiceWindow()
    w.add_interval(0, 10)
    w.add_interval(11, 12)
    w.add_interval(14, 16)

    assert w.contains(2)
    assert not w.contains(22)
    assert not w.contains(10.5)
    assert w.contains(9.5)
    assert not w.contains(16.5)


def test_window_to_juice_core() -> None:
    w = SpiceWindow()
    w.add_interval("2023-11-02T12:00:00.000", "2023-11-02T12:00:01.002")
    import os
    import tempfile
    from pathlib import Path

    tmpfile = tempfile.gettempdir() + "/test.csv"

    if Path(tmpfile).exists():
        os.remove(tmpfile)

    w.to_juice_core_csv(tmpfile)
    assert Path(tmpfile).exists()


def test_window_plot() -> None:
    """Shallow test to verify plot can be called"""
    w = SpiceWindow()
    w.add_interval(0, 10)
    w.add_interval(20, 30)

    got = w.plot()

    assert len(got) == len(w)

    w.plot(label="lab1")


def test_spice_window_from_to() -> None:
    from datetimerange import DateTimeRange
    from pandas import Timestamp
    from planetary_coverage import et

    s, e = tc.coverage

    s = Timestamp(s)
    e = Timestamp(e)

    trange = DateTimeRange(s, e)

    w = SpiceWindow.from_datetimerange([trange])

    assert w.start == et(tc.coverage[0])
    assert w.end == et(tc.coverage[1])

    asrange = w.to_datetimerange()[0]

    assert trange == asrange

    assert len(w) == 1

    with pytest.raises(ValueError):
        r = DateTimeRange(None, None)
        SpiceWindow.from_datetimerange([r])


def test_filters_fill() -> None:
    w1 = SpiceWindow()
    w1.add_interval(0, 10)
    w1.add_interval(11, 30)

    w1.fill_small_gaps(2)
    assert len(w1) == 1

    w1.add_interval(31, 31.1)
    w1.remove_small_intervals(0.2)
    assert len(w1) == 1


def test_conversions() -> None:
    w1 = SpiceWindow()
    w1.add_interval("2022-12-1T12:22:12.122", "2022-12-2")
    w2 = SpiceWindow()
    w2.add_interval("2023-12-1", "2023-12-2T12:02:11.0012")

    w3 = w1 + w2

    i1 = w3.to_datetimerange()[0]
    i2 = w3.to_datetimerange()[1]

    assert len(w3) == 2

    assert i1.start_datetime is not None
    assert i2.end_datetime is not None

    assert i1.start_datetime.year == 2022
    assert i2.end_datetime.year == 2023

    asdatetimerange = w3.to_datetimerange()

    new_window = SpiceWindow.from_datetimerange(asdatetimerange)
    assert w3.compare(new_window, operator="=")

    for i, interval in enumerate(w3):
        assert interval.start == new_window[i].start
        assert interval.end == new_window[i].end

    aspd = w3.to_pandas(round_to="ms")

    for i, interval in enumerate(w3):
        assert utc(interval.start) == aspd.iloc[i].start
        assert utc(interval.end) == aspd.iloc[i].end
