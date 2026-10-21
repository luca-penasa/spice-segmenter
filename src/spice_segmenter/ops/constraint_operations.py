from collections.abc import Iterable
from typing import Any

import pint
from anytree import Node
from loguru import logger as log

from ..core.constraints import ConstraintBase, ConstraintTypes
from ..core.property import Property
from ..properties.observation_properties import (
    MinMaxConditionTypes,
)
from ..support.decorators import vectorize
from ..support.model import segmenter_dataclass
from ..support.time_types import TIMES_TYPES
from ..support.types import PropertyNode


@segmenter_dataclass
class WrappedConstraint(ConstraintBase):
    parent: PropertyNode

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

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> bool:
        return self.parent(time)


@segmenter_dataclass
class Inverted(WrappedConstraint):
    def __repr__(self) -> str:
        return "NOT " + super().__repr__()

    def config(self, config: dict) -> None:
        super().config(config)
        config["inverted"] = True

    def tree(self) -> Node:
        return Node("NOT", children=[self.parent.tree()])

    @vectorize
    def __call__(self, time: TIMES_TYPES) -> bool:
        return ~self.parent(time)


@segmenter_dataclass
class MinMaxConstraint(ConstraintBase):
    wrapped_property: PropertyNode
    minmax_type: MinMaxConditionTypes
    adjust: float = 0.0  # this should be a pint quantity

    def __post_init__(self) -> None:
        if "local" in self.minmax_type.value:
            if self.adjust != 0.0:
                log.warning(
                    "Local minima and maxima cannot be adjusted, provided value {} will set to 0.0",
                    self.adjust,
                )
                self.adjust = 0.0

    def __call__(self, time: TIMES_TYPES) -> bool:
        raise NotImplementedError("MinMaxConstraints cannot be evaluated as a boolean")

    def __repr__(self) -> str:
        s = f"{self.wrapped_property} is a {self.name}"
        if "global" in self.minmax_type.value:
            s += f" within {self.adjust} threshold"

        s = f"({s})"
        return s

    def config(self, config: dict) -> None:
        self.wrapped_property.config(config)
        config["operator"] = self.minmax_type.value
        config["adjust"] = self.adjust

    @property
    def ctype(self) -> ConstraintTypes:
        return ConstraintTypes.MINMAX

    @property
    def right(self) -> Property:
        raise KeyError("MinMaxConstraints do not have a right side")

    @property
    def left(self) -> Property | ConstraintBase:
        return self.wrapped_property

    @property
    def operator(self) -> str:
        return self.minmax_type.value

    @property
    def name(self) -> str:
        return self.minmax_type.name

    @property
    def unit(self) -> Any | Iterable:
        return pint.Unit.dimensionless
