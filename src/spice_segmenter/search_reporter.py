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
        self.last_value = 0.0
        self.end_time = 0.0
        self.start_time = 0.0

    @property
    def update_function(self) -> Callable[[float, float, float], None]:
        def update_progress_report(istart: float, iend: float, et: float) -> None:
            interval_range = iend - istart

            percent = (et - istart) / interval_range * 100

            progress = percent - self.last_value
            self.last_value = percent
            self.bar.update(progress)

        return update_progress_report

    @property
    def update_function_spice(self) -> UDREPU:
        return spiceypy.utils.callbacks.SpiceUDREPU(self.update_function)

    @property
    def init_search(self) -> Callable[[SpiceCell, str, str], None]:
        def init_search(cell: SpiceCell, pre: str, suf: str) -> None:
            self.bar = tqdm(total=100, unit="%", desc=pre)
            self.reset()
            log.debug("Starting {}", pre)
            self.start_time = time.time()

        return init_search

    @property
    def init_search_spice(self) -> UDREPI:
        return spiceypy.utils.callbacks.SpiceUDREPI(self.init_search)

    @property
    def end_search(self) -> Callable[[], None]:
        def end_search() -> None:
            log.debug("Finished search!")
            self.end_time = time.time()

            log.debug("Time elapsed: {}", round(self.end_time - self.start_time, 2))
            self.bar.close()

        return end_search

    @property
    def end_search_spice(self) -> UDREPF:
        return spiceypy.utils.callbacks.SpiceUDREPF(self.end_search)


@define(repr=False, order=False, eq=False)
class NoSearchReporter:
    def reset(self) -> None:
        pass

    @property
    def update_function(self) -> Callable[[float, float, float], None]:
        def update_progress_report(istart: float, iend: float, et: float) -> None:
            pass

        return update_progress_report

    @property
    def update_function_spice(self) -> UDREPU:
        return spiceypy.utils.callbacks.SpiceUDREPU(self.update_function)

    @property
    def init_search(self) -> Callable[[SpiceCell, str, str], None]:
        def init_search(cell: SpiceCell, pre: str, suf: str) -> None:
            pass

        return init_search

    @property
    def init_search_spice(self) -> UDREPI:
        return spiceypy.utils.callbacks.SpiceUDREPI(self.init_search)

    @property
    def end_search(self) -> Callable[[], None]:
        def end_search() -> None:
            pass

        return end_search

    @property
    def end_search_spice(self) -> UDREPF:
        return spiceypy.utils.callbacks.SpiceUDREPF(self.end_search)


def get_default_reporter_class() -> SearchReporter | NoSearchReporter:
    from spice_segmenter import config

    if config["SHOW_PROGRESSBAR"]:
        return SearchReporter()
    else:
        return NoSearchReporter()
