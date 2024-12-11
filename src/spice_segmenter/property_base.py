
from typing import TYPE_CHECKING
from __future__ import annotations

from spice_segmenter.types import TIMES_TYPES

if TYPE_CHECKING:
    from spice_segmenter.constraint import Constraint, left_types
    from spice_segmenter.unit_adaptor import UnitAdaptor

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum, auto

import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define
from loguru import logger as log
from spiceypy.utils.callbacks import UDFUNB, UDFUNS


class PropertyTypes(Enum):
    SCALAR = auto()
    BOOLEAN = auto()
    VECTOR = auto()
    DISCRETE = auto()


@define(repr=False, order=False, eq=False)
class Property(ABC):
    @abstractmethod
    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum: ...

    def __str__(self) -> str:
        return self.__repr__()

    @property
    @abstractmethod
    def unit(self) -> pint.Unit | Iterable[pint.Unit]: ...

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.SCALAR

    def as_unit(self, unit: pint.Unit | str) -> "UnitAdaptor":
        from spice_segmenter.unit_adaptor import UnitAdaptor
        return UnitAdaptor(self, unit)

    def has_unit(self) -> bool:
        return bool(str(self.unit))

    @property
    @abstractmethod
    def name(self) -> str: ...

    def compute_as_spice_function(self) -> UDFUNS:
        def as_function(time: TIMES_TYPES) -> float | bool | Enum:
            return self.__call__(time)

        # TODO we are marking as_function as returing float, bool or enum, but wont work with SpiceUDFUNS. You need to use SpiceUDFUNB instead for booleans
        # while for enum wont work at all. Move these routines in the derived ScalarProperty and BooleanProperty classes please!
        return spiceypy.utils.callbacks.SpiceUDFUNS(as_function)

    def is_decreasing(self, time: TIMES_TYPES) -> bool:
        return spiceypy.uddc(self.compute_as_spice_function(), time, self.dt)  # type: ignore

    def is_decreasing_as_spice_function(self) -> UDFUNB:
        def as_function(function: Callable, time: TIMES_TYPES) -> bool:
            return self.is_decreasing(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)

    def __repr__(self) -> str:
        return f"{self.name}"

    def _handle_other_operand(self, other: "left_types") -> "Property":
        if isinstance(other, Property):
            return other

        from spice_segmenter.constant import Constant

        return Constant.from_value(other)

    def __gt__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        return Constraint(self, self._handle_other_operand(other), ">")

    def __ge__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        log.warning("Using >= operator on properties is not supported by SPICE. Using > instead.")
        return Constraint(self, self._handle_other_operand(other), ">")

    def __le__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        log.warning("Using <= operator on properties is not supported by SPICE. Using < instead.")
        return Constraint(self, self._handle_other_operand(other), "<")

    def __lt__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        return Constraint(self, self._handle_other_operand(other), "<")

    def __and__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        return Constraint(self, self._handle_other_operand(other), "&")

    def __eq__(self, other: "left_types") -> "Constraint":  # type: ignore
        other = self._handle_other_operand(other)
        if not isinstance(other, Property):
            return NotImplemented

        from spice_segmenter.constraint import Constraint

        return Constraint(self, other, "=")

    def __or__(self, other: "left_types") -> "Constraint":
        from spice_segmenter.constraint import Constraint
        return Constraint(self, self._handle_other_operand(other), "|")

    def config(self, config: dict) -> None:
        log.debug("adding prop unit for {}", self.unit)
        config["property_unit"] = str(self.unit)
        config["property"] = self.name


@define(repr=False, order=False, eq=False)
class BooleanProperty(Property):
    @abstractmethod
    def __call__(self, time: TIMES_TYPES) -> bool:
        pass

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("")

    def has_unit(self) -> bool:
        return False

    def compute_as_spice_function(self, invert: bool = False) -> UDFUNB:
        if invert:

            def as_function(udfun, time: TIMES_TYPES) -> bool:
                return ~self.__call__(time)

        else:

            def as_function(udfun, time: TIMES_TYPES) -> bool:
                return self.__call__(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)
