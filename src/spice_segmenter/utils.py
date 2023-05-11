from planetary_coverage.spice.times import et as _et

from spice_segmenter.types import times_types


def et(time: times_types) -> float:
    return _et(time)  # type: ignore
