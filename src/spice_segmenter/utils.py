from planetary_coverage.spice.times import et as _et

from spice_segmenter.types import TIMES_TYPES


def et(time: TIMES_TYPES) -> float:
    return _et(time)  # type: ignore
