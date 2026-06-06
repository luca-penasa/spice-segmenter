from __future__ import annotations

import operator as _op
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
from anytree import Node, RenderTree
from attrs import define, field
from loguru import logger as log

from spice_segmenter.properties.observation_properties import MinMaxConditionTypes

from .property import Property, PropertyTypes
from .time_segments_collection import TimeSegmentsCollection

if TYPE_CHECKING:
    from ..ops.constraint_operations import Inverted
    from ..support.time_types import TIMES_TYPES


left_types = Union["Property", str, float, int, Enum]

_OPERATOR_FNS = {
    ">": _op.gt,
    "<": _op.lt,
    "=": _op.eq,  # stored form produced by Property.__eq__
    "==": _op.eq,
    "&": np.logical_and,
    "|": np.logical_or,
}


class ConstraintTypes(Enum):
    COMPARE_TO_CONSTANT = auto()
    COMPARE_TO_OTHER_CONSTRAINT = auto()
    MINMAX = auto()


@define(repr=False, order=False, eq=False)
class ConstraintBase(Property):
    time_step: float | None = field(
        default=None,
        kw_only=True,
        converter=lambda x: (
            x if isinstance(x, float | None) else pd.Timedelta(x).total_seconds()
        ),
    )  # seconds

    @property
    @abstractmethod
    def left(self) -> Property | ConstraintBase: ...

    @property
    @abstractmethod
    def right(self) -> Property | ConstraintBase: ...

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
            log.error(
                "Cannot serialize a constraint that compares to another constraint",
            )
            raise TypeError

        self.left.config(config)
        self.right.config(config)
        config["operator"] = self.operator

    def get_compute_unit(self) -> pint.Unit | tuple | None:
        """Get the compute unit (native unit) for the left property via engine registry.

        Currently returns None to avoid circular imports during constraint initialization.
        To be implemented after solver refactoring.
        """
        # TODO: implement after solver refactor to access engine without circular imports
        return None

    def convert_reference_value(self, refval: float) -> float:
        """Convert a reference value from the constraint's desired unit to compute unit.

        Currently a no-op; to be implemented after solver refactoring.
        """
        # TODO: implement conversion logic after solver fully refactor
        return refval

    def solve(
        self,
        arg: Any = None,
        end: Any = None,
        **kwargs,
    ) -> TimeSegmentsCollection:
        """
        Solve the constraint over a time window.

        Parameters
        ----------
        arg : TimeSegmentsCollection or pd.Timestamp or str or float or int or tuple or object, optional
            One of:

            * A :class:`~spice_segmenter.core.TimeSegmentsCollection` — used directly as the search window.
            * A time-like start value (``str``, ``pd.Timestamp``, ``float`` ET, ``int`` ET) — must be
              paired with *end* or a second positional argument.
            * A 2-tuple ``(start, end)`` of time-like values.
            * Any object with ``start`` and ``end`` attributes.
            * Omitted — uses the default window from the active
              :class:`~spice_segmenter.support.config.Config` context (raises ``RuntimeError`` if none
              is set).
        end : time-like, optional
            End of the search window. Only used when *arg* is a time-like start value.
        optimize : bool, optional
            If True, apply constraint optimizations before solving.
            Default is False.
        **kwargs
            Additional keyword arguments passed to MasterSolver.

        Returns
        -------
        TimeSegmentsCollection
            Window containing intervals where the constraint is satisfied.

        Examples
        --------
        >>> constraint.solve(
        ...     TimeSegmentsCollection.from_start_end(
        ...         "2033-01-01",
        ...         "2033-12-31",
        ...     )
        ... )
        >>> constraint.solve(
        ...     "2033-01-01",
        ...     "2033-12-31",
        ... )  # two positional strings
        >>> constraint.solve(
        ...     (
        ...         "2033-01-01",
        ...         "2033-12-31",
        ...     )
        ... )  # tuple
        >>> constraint.solve(
        ...     et_start, et_end
        ... )  # SPICE ET floats
        >>> constraint.solve(
        ...     scenario
        ... )  # object with .start / .end
        >>> constraint.solve(
        ...     window,
        ...     optimize=True,
        ... )  # enable optimizer
        >>> with Config(
        ...     start="2033-01-01",
        ...     end="2033-12-31",
        ... ):
        ...     constraint.solve()  # uses default window from context
        """
        from ..support.config import get_active_config

        if end is not None:
            # Two-positional-arg form: solve(start, end)
            return self._solve_dispatch(arg, end, **kwargs)

        if arg is None:
            cfg = get_active_config()
            if cfg.start is None or cfg.end is None:
                raise RuntimeError(
                    "solve() called without a window and no default window is set. "
                    "Either pass a window explicitly or activate a context:\n\n"
                    "    with Config(start='2032-01-01', end='2035-01-01'):\n"
                    "        constraint.solve()\n",
                )
            arg = TimeSegmentsCollection.from_start_end(cfg.start, cfg.end)

        return self._solve_dispatch(arg, **kwargs)

    @singledispatchmethod
    def _solve_dispatch(self, arg: Any, **kwargs) -> TimeSegmentsCollection:
        """Internal dispatcher — do not call directly, use solve()."""
        optimize = kwargs.pop("optimize", False)

        if hasattr(arg, "start") and hasattr(arg, "end"):
            window = TimeSegmentsCollection.from_start_end(arg.start, arg.end)
            return self._solve_dispatch(window, optimize=optimize, **kwargs)

        raise TypeError(
            f"solve() called with unsupported type: {type(arg)}. "
            "Expected TimeSegmentsCollection, str, pd.Timestamp, or object with "
            "start/end attributes",
        )

    @_solve_dispatch.register
    def _(self, window: TimeSegmentsCollection, **kwargs) -> TimeSegmentsCollection:
        """Solve with a pre-constructed TimeSegmentsCollection."""
        from ..constraint_solver.constraint_solver import MasterSolver

        optimize = kwargs.pop("optimize", False)

        constraint = self
        if optimize:
            from ..optimizers.constraint_optimizer import optimize_constraint

            constraint = optimize_constraint(self, verbose=True)

        solver = MasterSolver(constraint=constraint, **kwargs)
        return solver.solve(window)

    @_solve_dispatch.register(str)
    @_solve_dispatch.register(pd.Timestamp)
    @_solve_dispatch.register(float)
    @_solve_dispatch.register(int)
    def _(
        self,
        start: pd.Timestamp | str | float,
        end: pd.Timestamp | str | float,
        **kwargs,
    ) -> TimeSegmentsCollection:
        """Solve by constructing a TimeSegmentsCollection from start/end times."""
        window = TimeSegmentsCollection.from_start_end(start, end)
        return self._solve_dispatch(window, **kwargs)

    @_solve_dispatch.register(tuple)
    def _(self, arg: tuple, **kwargs) -> TimeSegmentsCollection:
        """Solve from a 2-tuple (start, end) of time-like values."""
        if len(arg) != 2:
            raise ValueError(
                f"solve() expects a 2-tuple (start, end), got {len(arg)}-tuple.",
            )
        start, end = arg
        window = TimeSegmentsCollection.from_start_end(start, end)
        return self._solve_dispatch(window, **kwargs)

    def __invert__(self) -> Inverted:
        from ..ops.constraint_operations import Inverted

        return Inverted(self)

    def tree(self) -> Node:
        """
        Returns an anynode tree with the constraints
        """
        if isinstance(self.left, ConstraintBase) and isinstance(
            self.right,
            ConstraintBase,
        ):
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
        from ..ops.constant_values import Constant

        if not isinstance(self.right, Property):
            log.debug(
                "Left side of constraint {} is not a constraint or property. Assuming is a constant.",
                self,
            )
            self.right = Constant.from_value(self.right)

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

        if hasattr(self.right, "value") and isinstance(
            self.right.value,
            MinMaxConditionTypes,
        ):
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
            msg = f"Cannot Create a constraints between two properties with incompatible units: {self.left.unit} vs {self.right.unit}"
            log.warning(msg)
            # raise ValueError(msg)

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    @property
    def ctype(self) -> ConstraintTypes:
        from ..ops.constant_values import Constant

        if isinstance(self.right, Constant) or isinstance(self.left, Constant):
            return ConstraintTypes.COMPARE_TO_CONSTANT
        if isinstance(self.left, ConstraintBase) and isinstance(
            self.left,
            ConstraintBase,
        ):
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
        left_val = self.left(time)
        right_val = self.right(time)

        if self.left.unit != self.right.unit:
            log.warning(
                "Comparing {} with {}. This is not recommended. Will attempt automatic conversion.",
                self.left.unit,
                self.right.unit,
            )
            right_unit = self.right.unit
            if isinstance(right_unit, tuple):
                right_val = np.stack(
                    [
                        pint.Quantity(right_val[..., i], u).to(self.left.unit).magnitude
                        for i, u in enumerate(right_unit)
                    ],
                    axis=-1,
                )
            else:
                right_val = (
                    pint.Quantity(right_val, right_unit).to(self.left.unit).magnitude
                )

        fn = _OPERATOR_FNS.get(self.operator)
        if fn is None:
            raise ValueError(f"Unknown operator: {self.operator!r}")
        return np.array(fn(left_val, right_val), dtype=bool)
