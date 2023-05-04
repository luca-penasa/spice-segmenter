import time
from typing import Callable

import spiceypy
import spiceypy.utils.callbacks
from attr import define, field
from loguru import logger as log
from spiceypy import SpiceCell
from spiceypy.utils.callbacks import UDREPF, UDREPI, UDREPU
from tqdm.auto import tqdm
from tqdm.std import tqdm as std_tqdm


@define(repr=False, order=False, eq=False)
class SearchReporter:
    bar: std_tqdm = field(init=False, default=None)

    last_value: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def reset(self) -> None:
        if self.bar:
            self.bar.reset()
        self.last_value = 0.0
        self.end_time = 0.0
        self.start_time = 0.0

    @property
    def update_function(self) -> Callable[[float, float, float], None]:
        def update_progress_report(istart: float, iend: float, et: float) -> None:
            interval_range = iend - istart

            percent = (et - istart) / interval_range * 100
            # self.bar.update(percent)

            progress = percent - self.last_value
            self.last_value = percent
            self.bar.update(progress)
            # self.bar.n = percent
            # self.bar.last_print_n = percent
            # self.bar.refresh()

        return update_progress_report

    @property
    def update_function_spice(self) -> UDREPU:  # type: ignore
        return spiceypy.utils.callbacks.SpiceUDREPU(self.update_function)

    @property
    def init_search(self) -> Callable[[SpiceCell, str, str], None]:
        def init_search(cell: SpiceCell, pre: str, suf: str) -> None:
            self.bar = tqdm(total=100, unit="%", desc=pre)
            log.debug("Starting %s", pre)
            self.start_time = time.time()

        return init_search

    @property
    def init_search_spice(self) -> UDREPI:  # type: ignore
        return spiceypy.utils.callbacks.SpiceUDREPI(self.init_search)

    @property
    def end_search(self) -> Callable[[], None]:
        def end_search() -> None:
            log.debug("Finished search!")
            self.end_time = time.time()

            log.debug("Time elapsed: %s", round(self.end_time - self.start_time, 2))
            self.bar.close()

        return end_search

    @property
    def end_search_spice(self) -> UDREPF:  # type: ignore
        return spiceypy.utils.callbacks.SpiceUDREPF(self.end_search)
