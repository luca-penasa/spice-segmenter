from quick_spice_manager import QuickSpiceManager

tour_config = QuickSpiceManager(version="v432_20230505_001", mk="plan").get_tour_config(
    instrument=None,
)
