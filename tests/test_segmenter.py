from spice_segmenter.spice_window import SpiceWindow


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


def test_filters_fill() -> None:
    w1 = SpiceWindow()
    w1.add_interval(0, 10)
    w1.add_interval(11, 30)

    w1.fill_small_gaps(2)
    assert len(w1) == 1

    w1.add_interval(31, 31.1)
    w1.remove_small_intervals(0.2)
    assert len(w1) == 1


# from jana.spice_manger import SpiceManager

# man = SpiceManager()
# man.tour_config.load_kernels()

# def test_conversions():
#     w1 = SpiceWindow()
#     w1.add_interval("2022-12-1", "2022-12-2" )
#     w2 = SpiceWindow()
#     w2.add_interval("2023-12-1", "2023-12-2")

#     w3 = w1 + w2

#     assert w3.to_datetimerange()[0].start_datetime.year == 2022
#     assert w3.to_datetimerange()[1].end_datetime.year == 2023

#     asdatetimerange = w3.to_datetimerange()

#     new_window = SpiceWindow.from_datetimerange(asdatetimerange)
#     assert w3.compare(new_window, operator="=")
