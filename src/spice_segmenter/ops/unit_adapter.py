from __future__ import annotations

from typing import TYPE_CHECKING

import pint
from attrs import define, field

from ..core.property import Property
from ..support.decorators import vectorize
from ..support.spice_utilities import as_pint_unit

if TYPE_CHECKING:
    from ..support.time_types import TIMES_TYPES


@define(repr=False, order=False, eq=False)
class UnitAdaptor(Property):
    parent: Property
    _unit: pint.Unit = field(converter=as_pint_unit)

    @property
    def name(self) -> str:
        return self.parent.name

    @property
    def target(self) -> str:
        """Return a target if the property has one"""
        return self.parent.target

    @target.setter
    def target(self, target):
        self.parent.target = target

    @property
    def observer(self) -> str:
        """Return an observer if the property has one"""
        return self.parent.observer

    @observer.setter
    def observer(self, observer):
        self.parent.observer = observer

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
