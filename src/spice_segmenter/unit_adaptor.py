from __future__ import annotations

from typing import TYPE_CHECKING

import pint
from attrs import define, field

from spice_segmenter.decorators import vectorize
from spice_segmenter.property_base import Property
from spice_segmenter.utils import as_pint_unit

if TYPE_CHECKING:
    from spice_segmenter.types import TIMES_TYPES


@define(repr=False, order=False, eq=False)
class UnitAdaptor(Property):
    parent: Property
    _unit: pint.Unit = field(converter=as_pint_unit)

    @property
    def name(self) -> str:
        return self.parent.name

    @property
    def unit(self) -> pint.Unit:
        return self._unit

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float:
        return (  # type: ignore
            pint.Quantity(self.parent(time), self.parent.unit).to(self.unit).magnitude
        )

    def config(self, config: dict) -> None:
        return self.parent.config(config)
