from __future__ import annotations


from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import pint
from anytree import Node, RenderTree
from attrs import define, field
from loguru import logger as log


from spice_segmenter.property_base import Property, PropertyTypes
from spice_segmenter.spice_window import SpiceWindow
from spice_segmenter.trajectory_properties import MinMaxConditionTypes
from spice_segmenter.unit_adaptor import UnitAdaptor

if TYPE_CHECKING:
    from spice_segmenter.ops import Inverted
    from spice_segmenter.types import TIMES_TYPES


left_types = Union["Property", str, float, int, Enum]

class ConstraintTypes(Enum):
    COMPARE_TO_CONSTANT = auto()
    COMPARE_TO_OTHER_CONSTRAINT = auto()
    MINMAX = auto()


@define(repr=False, order=False, eq=False)
class ConstraintBase(Property):
    @property
    @abstractmethod
    def left(self) -> Property | "ConstraintBase": ...

    @property
    @abstractmethod
    def right(self) -> Property | "ConstraintBase": ...

    @property
    @abstractmethod
    def operator(self) -> str: ...

    @property
    @abstractmethod
    def ctype(self) -> ConstraintTypes: ...

    @abstractmethod
    def __call__(self, time: TIMES_TYPES) -> bool: ...

    def config(self, config: dict) -> None:
        if self.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            log.error("Cannot serialize a constraint that compares to another constraint")
            raise TypeError

        self.left.config(config)
        self.right.config(config)
        config["operator"] = self.operator

    def solve(self, window: SpiceWindow, **kwargs) -> SpiceWindow:  # type: ignore
        from .constraint_solver import MasterSolver

        solver = MasterSolver(constraint=self, **kwargs)
        return solver.solve(window)

    def __invert__(self) -> Inverted:
        from spice_segmenter.ops import Inverted

        return Inverted(self)

    def tree(self) -> Node:
        """
        Returns an anynode tree with the constraints
        """
        if isinstance(self.left, ConstraintBase) and isinstance(self.right, ConstraintBase):
            if self.operator == "|":
                name = "OR"

            elif self.operator == "&":
                name = "AND"

            else:
                raise NotImplementedError

            node = Node(name, children=[self.left.tree(), self.right.tree()])

        else:
            node = Node(self)

        return node

    def render_tree_str(self) -> str:
        """
        Get the tree as str
        """
        out = ""
        for pre, _fill, node in RenderTree(self.tree()):
            out += f"{pre}{node.name}\n"

        return out

    def render_tree(self) -> None:
        """
        Print anytree tree
        """
        for pre, _fill, node in RenderTree(self.tree()):
            print(f"{pre}{node.name}")


@define(repr=False, order=False, eq=False)
class Constraint(ConstraintBase):
    left: Property | ConstraintBase
    right: Property | ConstraintBase
    operator: str = field(default=None)

    def __attrs_post_init__(self) -> None:
        log.debug("Checking constraint {} for compatibility", self)
        log.debug("Left type is {}", type(self.left))
        log.debug("Right type is {}", type(self.right))
        from spice_segmenter.constant import Constant
        if not isinstance(self.right, Property):
            log.debug(
                "Left side of constraint {} is not a constraint or property. Assuming is a constant.",
                self,
            )
            self.right = Constant.from_value(self.right)

        if not self.left.has_unit() and not self.right.has_unit():
            log.debug("Both sides of constraint {} have no units, skipping check", self)

        elif self.ctype is not ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT and not self.right.has_unit():
            log.warning(
                "Constraint {} compares {} to {}",
                self,
                self.left.unit,
                self.right.unit,
            )
            return

        if hasattr(self.right, "value") and isinstance(self.right.value, MinMaxConditionTypes):
            log.debug("Right side of constraint {} is a minmax condition", self)
            return

        if isinstance(self.left.unit, Iterable):
            log.warning(
                "Constraint {} has a left side with multiple units: {}. This is not supported",
                self,
                self.left.unit,
            )
            raise NotImplementedError

        if not self.left.unit.is_compatible_with(self.right.unit):
            raise ValueError(
                f"Cannot Create a constraints between two properties with incompatible units: {self.left.unit} != {self.right.unit}",
            )

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    @property
    def ctype(self) -> ConstraintTypes:
        from spice_segmenter.constant import Constant
        if isinstance(self.right, Constant) or isinstance(self.left, Constant):
            return ConstraintTypes.COMPARE_TO_CONSTANT
        if isinstance(self.left, ConstraintBase) and isinstance(self.left, ConstraintBase):
            return ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT

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

    def __call__(self, time: TIMES_TYPES) -> npt.NDArray[np.bool_]:
        right: Property | None = None

        if self.left.unit != self.right.unit:
            log.warning(
                "Comparing {} with {}. This is not recommended. Will attempt automatic conversion.",
                self.left.unit,
                self.right.unit,
            )

            right = UnitAdaptor(self.right, self.left.unit)

        else:
            right = self.right

        if right is None:  # this is added just to make flake8 aware we are actually using it in the eval below
            raise ValueError("Could not convert right side of constraint")

        if self.operator == "=":
            operator = "=="
        else:
            operator = self.operator

        # TODO: is thera a better way to do this?
        q = "self.left(time)" + operator + "right(time)"

        return np.array(eval(q), dtype=bool)
