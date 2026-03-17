from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pint
from attrs import define, field

from ..core.property import Property
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

    def __call__(self, time: TIMES_TYPES):  # type: ignore[override]
        """Evaluate the parent property and convert its unit(s) to *self.unit*.

        For scalar properties the parent's ``unit`` is a single
        :class:`pint.Unit` and the conversion is straightforward.  The parent
        is already vectorised over time, so no additional ``@vectorize``
        wrapper is needed.

        For vector properties (e.g. coordinate tuples such as
        :class:`~spice_segmenter.properties.coordinates.PlanetographicCoordinates`)
        the parent's ``unit`` is a *tuple* of units — one per component.
        Each component is converted individually and the results are stacked
        back into the same output shape, preserving any leading time axes.
        """
        value = self.parent(time)
        parent_unit = self.parent.unit

        if isinstance(parent_unit, tuple):
            # Vector coordinate property: convert each component independently.
            return np.stack(
                [
                    pint.Quantity(value[..., i], u).to(self.unit).magnitude
                    for i, u in enumerate(parent_unit)
                ],
                axis=-1,
            )

        return pint.Quantity(value, parent_unit).to(self.unit).magnitude

    def config(self, config: dict) -> None:
        return self.parent.config(config)
