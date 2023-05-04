from typing import Iterable, Union

from planetary_coverage.spice.times import et as _et


def et(time: Iterable[str | float | int] | Union[str, float, int]) -> float:
    return _et(time)  # type: ignore
