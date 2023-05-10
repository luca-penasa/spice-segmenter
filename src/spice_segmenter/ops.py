from enum import Enum
from typing import Any, Iterable

import pint
from anytree import Node
from attrs import define, field
from loguru import logger as log

from spice_segmenter.trajectory_properties import (
    ConstraintBase,
    ConstraintTypes,
    Property,
)
from spice_segmenter.types import times_types

from .trajectory_properties import (
    ConstraintBase,
    ConstraintTypes,
    MinMaxConditionTypes,
    Property,
)


@define(repr=False, order=False, eq=False)
class WrappedConstraint(ConstraintBase):
    parent: ConstraintBase

    @property
    def name(self) -> str:
        return f"{self.parent.name}"

    @property
    def unit(self) -> Any | Iterable:
        return self.parent.unit

    @property
    def left(self) -> Property | ConstraintBase:
        return self.parent.left

    @property
    def right(self) -> Property | ConstraintBase:
        return self.parent.right

    @property
    def operator(self) -> str:
        return self.parent.operator

    @property
    def ctype(self) -> ConstraintTypes:
        return self.parent.ctype

    def tree(self) -> Node:
        return Node("wrapped", children=[self.parent.tree()])

    def __call__(self, time):
        return self.parent(time)


@define(repr=False, order=False, eq=False)
class Inverted(WrappedConstraint):
    def __repr__(self) -> str:
        return "NOT " + super().__repr__()

    def config(self, config: dict) -> None:
        super().config(config)
        config["inverted"] = True

    def tree(self) -> Node:
        return Node("NOT", children=[self.parent.tree()])

    def __call__(self, time):
        return ~self.parent(time)


# __call__, ctype, left, name, operator, right, unit


@define(repr=False, order=False, eq=False)
class MinMaxConstraint(ConstraintBase):
    property: Property
    minmax_type: MinMaxConditionTypes = field(converter=MinMaxConditionTypes)
    adjust: float = 0.0  # this should be a pint quantity

    def __attrs_post_init__(self) -> None:
        if "local" in self.minmax_type.value:
            if self.adjust != 0.0:
                log.warning(
                    "Local minima and maxima cannot be adjusted, provided value {} will set to 0.0",
                    self.adjust,
                )
                self.adjust = 0.0

    def __call__(self, time: times_types) -> float | bool | Enum:
        raise NotImplementedError

    def __repr__(self) -> str:
        s = f"{self.property} is a {self.name}"
        if "global" in self.minmax_type.value:
            s += f" within {self.adjust} threshold"

        s = f"({s})"
        return s

    def config(self, config: dict) -> None:
        self.property.config(config)
        config["operator"] = self.minmax_type.value
        config["adjust"] = self.adjust

    @property
    def ctype(self) -> ConstraintTypes:
        return ConstraintTypes.MINMAX

    @property
    def right(self) -> Property | ConstraintBase:
        return None

    @property
    def left(self) -> Property | ConstraintBase:
        return self.property

    @property
    def operator(self) -> str:
        return self.minmax_type.value

    @property
    def name(self) -> str:
        return self.minmax_type.name

    @property
    def unit(self) -> Any | Iterable:
        return pint.Unit.dimensionless
