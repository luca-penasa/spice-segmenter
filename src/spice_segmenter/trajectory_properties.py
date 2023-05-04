from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Iterable, Union

import pint
import spiceypy
import spiceypy.utils.callbacks
from anytree import Node, RenderTree
from attrs import define, field
from loguru import logger as log
from planetary_coverage.spice import SpiceRef
from spiceypy.utils.callbacks import UDFUNB, UDFUNS

from .decorators import vectorize
from .spice_window import SpiceWindow
from .types import times_types
from .utils import et

if TYPE_CHECKING:
    from spice_segmenter.occultation import OccultationTypes


left_types = Union["Property", str, float, int, Enum]


def as_spice_ref(item: str | int | SpiceRef) -> SpiceRef:
    if isinstance(item, SpiceRef):
        return item
    else:
        return SpiceRef(item)


def as_pint_unit(item: str | pint.Unit) -> pint.Unit:
    if isinstance(item, pint.Unit):
        return item
    else:
        return pint.Unit(item)


class PropertyTypes(Enum):
    SCALAR = auto()
    BOOLEAN = auto()
    VECTOR = auto()
    DISCRETE = auto()


@define(repr=False, order=False, eq=False)
class Property(ABC):
    @abstractmethod
    def __call__(self, time: times_types) -> float | bool | Enum:
        ...

    def __str__(self) -> str:
        return self.__repr__()

    @property
    @abstractmethod
    def unit(self) -> pint.Unit | Iterable[pint.Unit]:
        ...

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.SCALAR

    def as_unit(self, unit: pint.Unit | str) -> UnitAdaptor:
        return UnitAdaptor(self, unit)

    def has_unit(self) -> bool:
        return bool(str(self.unit))

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def compute_as_spice_function(self) -> UDFUNS:
        def as_function(time: times_types) -> float | bool | Enum:
            return self.__call__(time)

        return spiceypy.utils.callbacks.SpiceUDFUNS(as_function)

    def is_decreasing(self, time: times_types) -> bool:
        return spiceypy.uddc(self.compute_as_spice_function(), time, self.dt)  # type: ignore

    def is_decreasing_as_spice_function(self) -> UDFUNB:
        def as_function(function: Callable, time: times_types) -> bool:
            return self.is_decreasing(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)

    # def solve(self, window: SpiceWindow, relation: str, value: float):
    #     result = SpiceWindow(size=10000)
    #     return spiceypy.gfuds(
    #         self.compute_as_spice_function(),
    #         self.is_decreasing_as_spice_function(),
    #         relation,
    #         value,
    #         0.0,
    #         float(60 * 60),
    #         10000,
    #         window.spice_window,
    #         result.spice_window,
    #     )

    def __repr__(self) -> str:
        return f"{self.name}"

    def _handle_other_operand(self, other: left_types) -> Property:
        if isinstance(other, Property):
            return other
        else:
            return Constant(other)

    def __gt__(self, other: left_types) -> Constraint:
        return Constraint(self, self._handle_other_operand(other), ">")

    def __ge__(self, other: left_types) -> Constraint:
        log.warning(
            "Using >= operator on properties is not supported by SPICE. Using > instead."
        )
        return Constraint(self, self._handle_other_operand(other), ">")

    def __le__(self, other: left_types) -> Constraint:
        log.warning(
            "Using <= operator on properties is not supported by SPICE. Using < instead."
        )
        return Constraint(self, self._handle_other_operand(other), "<")

    def __lt__(self, other: left_types) -> Constraint:
        return Constraint(self, self._handle_other_operand(other), "<")

    def __and__(self, other: left_types) -> Constraint:
        return Constraint(self, self._handle_other_operand(other), "&")

    def __eq__(self, other: left_types) -> Constraint:  # type: ignore
        other = self._handle_other_operand(other)
        if not isinstance(other, Property):
            return NotImplemented

        return Constraint(self, other, "==")

    def __or__(self, other: left_types) -> Constraint:
        return Constraint(self, self._handle_other_operand(other), "|")

    def config(self, config: dict) -> None:
        log.debug("adding prop unit for %s", self.unit)
        config["property_unit"] = str(self.unit)


@define(repr=False, order=False, eq=False)
class Constant(Property):
    _value: pint.Quantity = field(converter=lambda x: pint.Quantity(x))  # type: ignore

    def __repr__(self) -> str:
        val = f"{self.value}"

        if str(self.unit):
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
    def __call__(self, time: times_types) -> float | OccultationTypes:
        return self._value.magnitude  # type: ignore

    def config(self, config: dict) -> None:
        config.update({"reference_value": self.value})
        config["reference_value_unit"] = str(self.unit)


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
    def __call__(self, time: times_types) -> float:
        return (  # type: ignore
            pint.Quantity(self.parent(time), self.parent.unit).to(self.unit).magnitude
        )

    def config(self, config: dict) -> None:
        return self.parent.config(config)


@define(repr=False, order=False, eq=False)
class TargetedProperty(Property, ABC):
    observer: SpiceRef = field(converter=as_spice_ref)
    target: SpiceRef = field(converter=as_spice_ref)
    light_time_correction: str = field(default="NONE", kw_only=True)

    def config(self, config: dict) -> None:
        log.debug(
            "targeted property config here with instnace of %s", self.__class__.__name__
        )
        Property.config(self, config)
        config.update(
            {
                "target": self.target.name,
                "observer": self.observer.name,
                "abcorr": self.light_time_correction,
            }
        )


@define(repr=False, order=False, eq=False)
class PhaseAngle(TargetedProperty):
    third_body: SpiceRef = field(
        factory=lambda: as_spice_ref("SUN"), converter=as_spice_ref  # type: ignore
    )

    def __repr__(self) -> str:
        return f"Phase Angle of {self.target} with respect to {self.third_body} as seen from {self.observer}"

    @property
    def name(self) -> str:
        return "phase_angle"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("rad")

    @vectorize
    def __call__(self, time: times_types) -> float:
        return spiceypy.phaseq(  # type: ignore
            et(time),
            self.target.name,
            self.third_body.name,
            self.observer.name,
            self.light_time_correction,
        )

    def config(self, config: dict) -> None:
        TargetedProperty.config(self, config)
        config["third_body"] = self.third_body.name
        config["property"] = self.name


@define(repr=False, order=False, eq=False)
class Distance(TargetedProperty):
    @property
    def name(self) -> str:
        return "distance"

    def __repr__(self) -> str:
        return f"Distance of {self.target} from {self.observer}"

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("km")

    @vectorize
    def __call__(self, time: times_types) -> float:
        return spiceypy.vnorm(  # type: ignore
            spiceypy.spkpos(
                self.target.name,
                et(time),
                self.observer.frame,  # type: ignore
                self.light_time_correction,
                self.observer.name,
            )[0]
        )

    def config(self, config: dict) -> None:
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
        if not self.left.has_unit() and not self.right.has_unit():
            log.debug("Both sides of constraint {} have no units, skipping check", self)

        elif (
            self.ctype is not ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT
            and not self.right.has_unit()
        ):
            log.warning(
                "Constraint {} compares {} to {}",
                self,
                self.left.unit,
                self.right.unit,
            )
            return

        if not self.left.unit.is_compatible_with(self.right.unit):
            raise ValueError(
                f"Cannot Create a constraints between two properties with incompatible units: {self.left.unit} != {self.right.unit}"
            )

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    def config(self, config: dict) -> None:
        if self.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            log.error(
                "Cannot serialize a constraint that compares to another constraint"
            )
            raise TypeError

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
        right = None
        if self.left.unit != self.right.unit:
            log.warning(
                "Comparing {} with {}. This is not recommended. Will attempt automatic conversion.",
                self.left.unit,
                self.right.unit,
            )

            right = UnitAdaptor(self.right, self.left.unit)

        else:
            right = self.right

        if (
            right is None
        ):  # this is added just to make flake8 aware we are actually using it in the eval below
            raise ValueError("Could not convert right side of constraint")

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
        for pre, _fill, node in RenderTree(self.tree()):
            print(f"{pre}{node.name}")

    def solve(self, window: SpiceWindow, **kwargs) -> SpiceWindow:  # type: ignore
        from .constraint_solver import MasterSolver

        solver = MasterSolver(self, **kwargs)
        return solver.solve(window)
