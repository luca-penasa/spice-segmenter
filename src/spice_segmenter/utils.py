import pint
from planetary_coverage.spice import SpiceRef
from planetary_coverage.spice.times import et as _et

from spice_segmenter.types import TIMES_TYPES


def et(time: TIMES_TYPES) -> float:
    return _et(time)  # type: ignore



def add_properties_to_table(tab, properties, observer):
    """Table rows must have start, end and target columns."""
    for i, row in tab.iterrows():
        for Prop in properties:
            try:
                aa = Prop(observer=observer, target= row.target)
            except:
                continue

            mid = row.start + (row.end - row.start)/2

            res = aa(str(mid.tz_localize(None)))

            name = f"{aa.name} [{aa.unit}] "
            tab.loc[i, name] = res


def as_pint_unit(item: str | pint.Unit) -> pint.Unit:
    if isinstance(item, pint.Unit):
        return item
    return pint.Unit(item)


def as_spice_ref(item: str | int | SpiceRef) -> SpiceRef:
    if isinstance(item, SpiceRef):
        return item
    return SpiceRef(item)
