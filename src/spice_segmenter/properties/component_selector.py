from __future__ import annotations

from typing import Any, ClassVar

import pint
from attr import define, field
from attrs.validators import instance_of

from ..core.property import Property, PropertyTypes
from ..support.decorators import vectorize
from ..support.time_types import TIMES_TYPES


@define(repr=False, order=False, eq=False)
class ComponentSelector(Property):
    # ComponentSelector is a helper/wrapper, not a standalone computable property.
    # Exclude it from compute_all auto-iteration.
    _skip_auto_compute: ClassVar[bool] = True

    vector: Property = field(default=None)
    component: int = field(default=0, converter=int)
    _name: str = "component_selector"
    _unit_override: pint.Unit | None = field(default=None)

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
        if self._unit_override is not None:
            return self._unit_override
        return self.vector.unit[self.component]

    def as_unit(self, unit: pint.Unit | str) -> "ComponentSelector":
        """Return a copy that evaluates in *unit*, applying conversion in __call__."""
        import attrs
        from ..core.property import _to_pint_unit

        target = _to_pint_unit(unit)
        current = self.vector.unit[self.component]
        if current is not None:
            if not current.is_compatible_with(target):
                raise ValueError(
                    f"{self!r}: unit {target} is not compatible with current unit {current}",
                )
        return attrs.evolve(self, unit_override=target)

    @vectorize()
    def __call__(self, time: TIMES_TYPES) -> float:
        value = self.vector.__call__(time)[self.component]
        if self._unit_override is not None:
            native = self.vector.unit[self.component]
            value = (value * native).to(self._unit_override).magnitude
        return value

    def config(self, config: dict) -> None:
        self.vector.config(config)
        config["component"] = self.name
        config["property_unit"] = str(self.unit)
