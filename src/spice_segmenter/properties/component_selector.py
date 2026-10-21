from __future__ import annotations

from typing import Annotated, Any, ClassVar

import pint
from pydantic import AfterValidator, Field

from ..core.property import Property, PropertyTypes
from ..core.registry import property_registry
from ..support.decorators import vectorize
from ..support.model import segmenter_dataclass
from ..support.time_types import TIMES_TYPES
from ..support.types import OptionalUnitField, PropertyNode, converted

_Index = converted(int)


def _check_vector(value: Any) -> Property:
    if not isinstance(value, Property):
        raise ValueError(f"vector must be a Property, got {value!r}")
    if not value.type == PropertyTypes.VECTOR:
        raise ValueError(f"Vector must be of type {PropertyTypes.VECTOR}")
    return value


@segmenter_dataclass
class ComponentSelector(Property):
    # ComponentSelector is a helper/wrapper, not a standalone computable property.
    # Exclude it from compute_all auto-iteration.
    _skip_auto_compute: ClassVar[bool] = True

    vector: Annotated[PropertyNode, AfterValidator(_check_vector)] = None
    component: _Index = 0
    _name: str = Field(default="component_selector", alias="name")
    _unit_override: OptionalUnitField = Field(default=None, alias="unit_override")

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
        from ..core.property import _to_pint_unit
        from ..support.fields import evolve

        target = _to_pint_unit(unit)
        current = self.vector.unit[self.component]
        if current is not None:
            if not current.is_compatible_with(target):
                raise ValueError(
                    f"{self!r}: unit {target} is not compatible with current unit {current}",
                )
        return evolve(self, _unit_override=target)

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


# ``_name`` is a field here (each selector names its component), so
# __init_subclass__, which registers classes by their string ``_name``, skips it.
property_registry.register("component_selector", ComponentSelector)
