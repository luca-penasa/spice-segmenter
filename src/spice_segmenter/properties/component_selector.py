from __future__ import annotations

from typing import Any

import pint
from attr import define, field
from attrs.validators import instance_of

from ..support.decorators import vectorize
from ..core.property import Property, PropertyTypes
from ..support.time_types import TIMES_TYPES


@define(repr=False, order=False, eq=False)
class ComponentSelector(Property):
    vector: Property = field(default=None)
    component: int = field(default=0, converter=int)
    _name: str = "component_selector"

    @vector.validator
    def _validate_vector(self, attribute, value) -> Any:  # type: ignore
        if not value.type == PropertyTypes.VECTOR:
            raise ValueError(f"Vector must be of type {PropertyTypes.VECTOR}")

        return instance_of(Property)(self, attribute, value)

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.SCALAR

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> pint.Unit:
        return self.vector.unit[self.component]

    @vectorize()
    def __call__(self, time: TIMES_TYPES) -> float:
        return self.vector.__call__(time)[self.component]

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["component"] = self.name
        config["property_unit"] = str(self.unit)
