from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pint
from attrs import define, field

from spice_segmenter.decorators import vectorize
from spice_segmenter.property_base import Property

if TYPE_CHECKING:
    from spice_segmenter.occultation import OccultationTypes
    from spice_segmenter.types import TIMES_TYPES

@define(repr=False, order=False, eq=False)
class Constant(Property):
    @staticmethod
    def from_value(value):
        if isinstance(value, bool):
            return BoolConstant(value)

        if isinstance(value, (int, float, str, Enum, pint.Quantity)):
            return ScalarConstant(value)

        raise NotImplementedError


@define(repr=False, order=False, eq=False)
class ScalarConstant(Constant):
    _value: pint.Quantity = field(converter=lambda x: pint.Quantity(x))  # type: ignore

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


@define(repr=False, order=False, eq=False)
class BoolConstant(Constant):
    _value: pint.Quantity = field(converter=bool)  # type: ignore

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


