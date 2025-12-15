from typing import TYPE_CHECKING, Union

import pint
from loguru import logger as log
from planetary_coverage.spice import SpiceRef
from planetary_coverage.spice.times import et as _et

if TYPE_CHECKING:
    from spice_segmenter.trajectory_properties import TargetedProperty

from spice_segmenter.property_base import Property
from spice_segmenter.types import TIMES_TYPES


def et(time: TIMES_TYPES) -> float:
    return _et(time)  # type: ignore


def add_properties_to_table(
    tab,
    properties: list[Union["TargetedProperty",  type["TargetedProperty"]]],
    observer: str,
    timecol=None,
    targetcol="target",
    *,
    retarget_instances=True,
):

    """Table rows must have start, end and target columns."""
    for i, row in tab.iterrows():
        target = row[targetcol]

        for property in properties:
            log.debug(f"Working on property {property}")
            if isinstance(property, Property):
                log.debug(f"found instance of property {property}")
                prop_instance = property
                if retarget_instances:
                    prop_instance.target = target

                # prop_instance.observer = observer
            else:
                log.debug("found class. Instantiation.")
                prop_instance = property(observer=observer, target=target)

            if timecol is None:
                ref_time = row.start + (row.end - row.start) / 2
            else:
                ref_time = row[timecol]

            res = prop_instance(str(ref_time.tz_localize(None)))

            name = f"{prop_instance.name} [{prop_instance.unit}] "
            tab.loc[i, name] = res


def as_pint_unit(item: str | pint.Unit) -> pint.Unit:
    if isinstance(item, pint.Unit):
        return item
    return pint.Unit(item)


def as_spice_ref(item: str | int | SpiceRef) -> SpiceRef:
    if isinstance(item, SpiceRef):
        return item
    return SpiceRef(item)
