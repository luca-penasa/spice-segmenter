from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pint
from pydantic import Field

from ..core.property import Property
from ..support.decorators import vectorize
from ..support.model import segmenter_dataclass
from ..support.types import converted

if TYPE_CHECKING:
    from ..properties.occultation_types import OccultationTypes
    from ..support.time_types import TIMES_TYPES

# Copy into a new quantity (as the attrs converter did): the unit setter of
# ScalarConstant mutates it in place.
_QuantityValue = converted(lambda x: pint.Quantity(x), serializer=str)
def _to_bool(x: object) -> bool:
    # Former serializations wrote booleans as "True" / "False".
    if isinstance(x, str) and x.strip().lower() in ("true", "false"):
        return x.strip().lower() == "true"
    return bool(x)


_BoolValue = converted(_to_bool)

@segmenter_dataclass
class Constant(Property):
    @staticmethod
    def from_value(value):
        if isinstance(value, bool):
            return BoolConstant(value)

        if isinstance(value, (int, float, str, Enum, pint.Quantity)):
            return ScalarConstant(value)

        raise NotImplementedError


@segmenter_dataclass
class ScalarConstant(Constant):
    _value: _QuantityValue = Field(alias="value")

    def __repr__(self) -> str:
        val = f"{self.value}"

        if str(self.unit) != "dimensionless":
            val += f" {self.unit}"

        return val

    @property
    def name(self) -> str:
        return "constant"

    @property
    def value(self) -> float | OccultationTypes:
        return self._value.magnitude  # type: ignore

    @property
    def unit(self) -> pint.Unit:
        return self._value.u

    @unit.setter
    def unit(self, unit: pint.Unit) -> pint.Unit:
        self._value.u = unit

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float | OccultationTypes:
        return self._value.magnitude  # type: ignore

    def config(self, config: dict) -> None:
        config.update({"reference_value": self.value})
        config["reference_value_unit"] = str(self.unit)


@segmenter_dataclass
class BoolConstant(Constant):
    _value: _BoolValue = Field(alias="value")

    def __repr__(self) -> str:
        return f"{self.value}"

    @property
    def name(self) -> str:
        return "bool_constant"

    @property
    def value(self) -> float | OccultationTypes:
        return self._value

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("")

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> float | OccultationTypes:
        return self._value  # type: ignore

    def config(self, config: dict) -> None:
        config.update({"reference_value": self.value})


