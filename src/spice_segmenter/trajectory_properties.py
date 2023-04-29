from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Iterable

import pint
import spiceypy
import spiceypy.utils.callbacks
from anytree import Node, RenderTree
from attrs import define, field
from loguru import logger as log
from planetary_coverage.spice import SpiceRef
from planetary_coverage.spice.times import et
from spiceypy.utils.callbacks import UDFUNS

from .decorators import vectorize
from .spice_window import SpiceWindow
from .types import obj_type, times_types

if TYPE_CHECKING:
    from spice_segmenter.occultation import OccultationTypes


class PropertyTypes(Enum):
    SCALAR = auto()
    BOOLEAN = auto()
    VECTOR = auto()
    DISCRETE = auto()


@define(repr=False, order=False, eq=False)
class Property(ABC):
    @abstractmethod
    def __call__(self, time: times_types) -> float | bool:
        ...

    @property
    @abstractmethod
    def unit(self) -> pint.Unit | Iterable[pint.Unit]:
        ...

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.SCALAR

    def as_unit(self, unit: pint.Unit | str):
        return UnitAdaptor(self, unit)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def compute_as_spice_function(self) -> UDFUNS:
        def as_function(time: times_types):
            return self.__call__(time)

        return spiceypy.utils.callbacks.SpiceUDFUNS(as_function)

    def is_decreasing(self, time: times_types) -> bool:
        return spiceypy.uddc(self.compute_as_spice_function(), time, self.dt)

    def is_decreasing_as_spice_function(self):
        def as_function(callable, time: times_types):
            return self.is_decreasing(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)

    def solve(self, window: SpiceWindow, relation: str, value: float):
        result = SpiceWindow(size=10000)
        return spiceypy.gfuds(
            self.compute_as_spice_function(),
            self.is_decreasing_as_spice_function(),
            relation,
            value,
            0.0,
            float(60 * 60),
            10000,
            window.spice_window,
            result.spice_window,
        )

    def __repr__(self):
        return f"{self.name}"

    def _handle_other_operand(self, other: Property) -> Property:
        if isinstance(other, Property):
            return other
        else:
            return Constant(other)

    def __gt__(self, other):
        return Constraint(self, self._handle_other_operand(other), ">")

    def __ge__(self, other):
        log.warning(
            "Using >= operator on properties is not supported by SPICE. Using > instead."
        )
        return Constraint(self, self._handle_other_operand(other), ">")

    def __le__(self, other):
        log.warning(
            "Using <= operator on properties is not supported by SPICE. Using < instead."
        )
        return Constraint(self, self._handle_other_operand(other), "<")

    def __lt__(self, other):
        return Constraint(self, self._handle_other_operand(other), "<")

    def __and__(self, other):
        return Constraint(self, self._handle_other_operand(other), "&")

    def __eq__(self, other):
        return Constraint(self, self._handle_other_operand(other), "==")

    def __or__(self, other):
        return Constraint(self, self._handle_other_operand(other), "|")

    def __ne__(self, other):
        return Constraint(self, self._handle_other_operand(other), "!=")

    def config(self, config: dict):
        log.debug(f"adding prop unit for {self.unit}")
        config["property_unit"] = str(self.unit)


@define(repr=False, order=False, eq=False)
class Constant(Property):
    _value: pint.Quantity = field(converter=pint.Quantity)

    def __repr__(self) -> str:
        val = f"{self.value}"

        if self.unit != pint.Unit(""):
            val += f" {self.unit}"

        return val

    @property
    def name(self) -> str:
        return "constant"

    @property
    def value(self) -> float | OccultationTypes:
        return self._value.magnitude

    @property
    def unit(self) -> pint.Unit:
        return self._value.u

    @unit.setter
    def unit(self, unit: pint.Unit):
        self._value.u = unit

    @vectorize
    def __call__(self, time: times_types) -> float | OccultationTypes:
        return self._value.magnitude

    def config(self, config: dict) -> None:
        config.update(dict(reference_value=self.value))
        config["reference_value_unit"] = str(self.unit)


@define(repr=False, order=False, eq=False)
class UnitAdaptor(Property):
    parent: Property
    _unit: pint.Unit = field(converter=pint.Unit)

    @property
    def name(self) -> str:
        return self.parent.name

    @property
    def unit(self) -> pint.Unit:
        return self._unit

    @vectorize
    def __call__(self, time: times_types) -> float:
        return (
            pint.Quantity(self.parent(time), self.parent.unit).to(self.unit).magnitude
        )

    def config(self, config: dict):
        return self.parent.config(config)


@define(repr=False, order=False, eq=False)
class TargetedProperty(Property, ABC):
    observer: obj_type = field(converter=SpiceRef)
    target: obj_type = field(converter=SpiceRef)
    light_time_correction: str = field(default="NONE")

    def config(self, config: dict) -> None:
        log.debug(
            f"targeted property config here with instnace of {self.__class__.__name__}"
        )
        Property.config(self, config)
        config.update(
            dict(
                target=self.target.name,
                observer=self.observer.name,
                abcorr=self.light_time_correction,
            )
        )
        return config


@define(repr=False, order=False, eq=False)
class PhaseAngle(TargetedProperty):
    third_body: SpiceRef = field(factory=lambda: SpiceRef("SUN"), converter=SpiceRef)

    @property
    def name(self) -> str:
        return f"phase_angle"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad")

    @vectorize
    def __call__(self, time: times_types) -> float:
        return spiceypy.phaseq(
            et(time),
            self.target.name,
            self.third_body.name,
            self.observer.name,
            self.light_time_correction,
        )

    def config(self, config: dict):
        TargetedProperty.config(self, config)
        config.update(dict(third_body=self.third_body.name))
        config["property"] = self.name


@define(repr=False, order=False, eq=False)
class Distance(TargetedProperty):
    @property
    def name(self) -> str:
        return "distance"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km")

    @vectorize
    def __call__(self, time: times_types) -> float:
        return spiceypy.vnorm(
            spiceypy.spkpos(
                self.target.name,
                et(time),
                self.observer.frame,
                self.light_time_correction,
                self.observer.name,
            )[0]
        )

    def config(self, config: dict):
        TargetedProperty.config(self, config)
        config["property"] = self.name


class ConstraintTypes(Enum):
    COMPARE_TO_CONSTANT = auto()
    COMPARE_TO_OTHER_CONSTRAINT = auto()


@define(repr=False, order=False, eq=False)
class Constraint(Property):
    left: Property | Constraint
    right: Property | Constraint
    operator: str = field(default=None)

    def __attrs_post_init__(self) -> None:
        if self.ctype is not ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT and (
            self.right.unit == pint.Unit("")
        ):
            log.warning("Comparing to dimenionless constant")
            return

        if not self.left.unit.is_compatible_with(self.right.unit):
            raise ValueError(
                f"Cannot Create a constraints between two properties with different units: {self.left.unit} != {self.right.unit}"
            )

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    def config(self, config: dict) -> None:
        if self.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            raise ValueError(
                "Cannot serialize a constraint that compares to another constraint"
            )

        self.left.config(config)
        self.right.config(config)
        config["operator"] = self.operator

    @property
    def ctype(self) -> ConstraintTypes:
        if Constant in self.types:
            return ConstraintTypes.COMPARE_TO_CONSTANT
        elif self.types[0] == Constraint and self.types[1] == Constraint:
            return ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT

        else:
            log.error("Cannot determine constraint type")
            raise NotImplementedError

    @property
    def types(self) -> tuple[type[Property], type[Property]]:
        return type(self.left), type(self.right)

    @property
    def name(self) -> str:
        return f"{self}"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("")  # a constraint has no unit, as it returns bools

    def __call__(self, time: times_types) -> bool:
        if self.left.unit != self.right.unit:
            log.warning(
                f"Comparing {self.left.unit} with {self.right.unit}. This is not recommended."
            )

            UnitAdaptor(self.right, self.left.unit)

        else:
            self.right

        q = "self.left(time)" + self.operator + "right(time)"

        return eval(q)

    def tree(self) -> Node:
        """
        Returns an anynode tree with the constraints
        """
        if isinstance(self.left, Constraint) and isinstance(self.right, Constraint):
            node = Node(self, children=[self.left.tree(), self.right.tree()])

        else:
            node = Node(self)

        return node

    def render_tree(self) -> None:
        """
        Print anytree tree
        """
        for pre, fill, node in RenderTree(self.tree()):
            print(f"{pre}{node.name}")

    def solve(self, window: SpiceWindow, **kwargs) -> SpiceWindow:
        from .constraint_solver import MasterSolver

        solver = MasterSolver(self, **kwargs)
        return solver.solve(window)
