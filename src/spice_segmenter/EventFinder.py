from datetime import datetime
from typing import Union

import numpy as np
import spiceypy
from attr import field
from datetimerange import DateTimeRange
from planetary_coverage import et

from .attrs_custom import define
from .trajectory_properties import Constraint

time_type = Union[float, str, np.datetime64]

from more_itertools import grouper


@define
class SpiceEventFinder:
    _constraint: Constraint = field(default=None)
    _start: float = field(default=None, converter=et)
    _end: float = field(default=None, converter=et)

    # qpnams = ["TARGET", "OBSERVER", "ABCORR"],
    # qcpars = ["JUPITER", "JUICE", "LT+S"],

    @property
    def start(self) -> datetime:
        return spiceypy.et2datetime(self._start)

    @property
    def end(self) -> datetime:
        return spiceypy.et2datetime(self._end)

    @property
    def window(self) -> DateTimeRange:
        return DateTimeRange(self.start, self.end)

    def _to_datetime_ranges(self, results):
        def as_range(start, end):
            start = spiceypy.et2datetime(start)
            end = spiceypy.et2datetime(end)
            return DateTimeRange(start, end)

        return [as_range(start, end) for start, end in grouper(results, 2)]

    def solve(self):
        pars = self._constraint.to_spice_gfevnt_config()

        maxval = 10000
        step = 0.01 * spiceypy.spd()
        cnfine = spiceypy.Cell_Double(maxval)  # the window
        result = spiceypy.Cell_Double(maxval)  # the resulting window

        spiceypy.wninsd(self._start, self._end, cnfine)  # add 1 window to the cell
        spiceypy.gfsstp(step)  # set the step size

        from spiceypy import gfrefn, gfstep

        spiceypy.gfevnt(
            udstep=spiceypy.utils.callbacks.SpiceUDFUNS(gfstep),
            udrefn=spiceypy.utils.callbacks.SpiceUDREFN(gfrefn),
            qnpars=len(pars["qpnams"]),
            lenvals=100,
            qdpars=[],
            qipars=[],
            qlpars=[],
            tol=1e-3,
            adjust=0.0,
            rpt=True,
            udrepi=spiceypy.utils.callbacks.SpiceUDREPI(spiceypy.gfrepi),
            udrepu=spiceypy.utils.callbacks.SpiceUDREPU(spiceypy.gfrepu),
            udrepf=spiceypy.utils.callbacks.SpiceUDREPF(spiceypy.gfrepf),
            nintvls=maxval,
            bail=True,
            udbail=spiceypy.utils.callbacks.SpiceUDBAIL(spiceypy.gfbail),
            cnfine=cnfine,
            result=result,
            **pars,
        )

        return self._to_datetime_ranges(result)
