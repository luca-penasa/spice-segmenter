"""Pydantic dataclass base for the segmenter's data classes.

Every data class of the package (properties, constraints, solvers, contexts,
time segments) is a :func:`pydantic.dataclasses.dataclass` created through
:func:`segmenter_dataclass`, which fixes the options the package relies on:

* ``eq=False`` by default: ``<``, ``>``, ``==``, ``&`` and ``|`` on properties
  build constraints, so dataclass equality must stay off;
* ``repr=False`` by default: properties define their own ``__repr__``;
* ``slots=True`` by default, as with attrs: a field declared in a subclass then
  overrides a ``@property`` of the same name on a base class (``unit``, the
  abstract ``left`` / ``right`` of :class:`ConstraintBase`, ...);
* :data:`MODEL_CONFIG`: arbitrary types (SPICE references, pint units, numpy
  arrays) are allowed, defaults are validated like explicit values (attrs ran
  converters on defaults too), assignments are converted like constructor
  arguments, unknown keyword arguments are rejected, and fields can be passed
  by name or by alias.

Fields with a conversion step (strings to SPICE references or pint units, ...)
use the annotated types of :mod:`spice_segmenter.support.types`.
"""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Callable
from typing import Any, TypeVar, dataclass_transform, overload

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as _pydantic_dataclass

T = TypeVar("T", bound=type)

MODEL_CONFIG = ConfigDict(
    # Unknown keyword arguments are errors (as with attrs), not silently dropped.
    extra="forbid",
    arbitrary_types_allowed=True,
    validate_default=True,
    # attrs ran converters on assignment too (``config.solver_step = 3600``).
    validate_assignment=True,
    validate_by_name=True,
    validate_by_alias=True,
)


def _rebind_class_cells(old: type, new: type) -> None:
    """Point ``__class__`` closure cells of *new*'s methods at *new*.

    Creating a slotted dataclass builds a new class object, while the methods
    still close over the original one, so zero-argument ``super()`` would fail
    on Python versions whose ``dataclasses`` does not fix this itself.
    """
    for value in new.__dict__.values():
        candidates: list[Any] = [value]
        if isinstance(value, (classmethod, staticmethod)):
            candidates = [value.__func__]
        elif isinstance(value, property):
            candidates = [value.fget, value.fset, value.fdel]
        for candidate in candidates:
            func = getattr(candidate, "__func__", candidate)
            func = getattr(func, "__wrapped__", func)
            if not isinstance(func, types.FunctionType) or not func.__closure__:
                continue
            for cell in func.__closure__:
                try:
                    contents = cell.cell_contents
                except ValueError:  # empty cell
                    continue
                if contents is old:
                    cell.cell_contents = new


def _shadow_inherited_properties(cls: type) -> None:
    """Mark fields that redefine an inherited ``@property`` as required.

    ``dataclasses`` reads a field's default with ``getattr(cls, name)``, which
    finds a base-class property of the same name (e.g. the abstract ``left`` /
    ``right`` of ``ConstraintBase``) and would take the property object as the
    default. An explicit ``Field()`` keeps such fields required.
    """
    for name, annotation in cls.__dict__.get("__annotations__", {}).items():
        if name in cls.__dict__ or "ClassVar" in str(annotation):
            continue
        if isinstance(inspect.getattr_static(cls, name, None), property):
            setattr(cls, name, Field())


@overload
def segmenter_dataclass(cls: T, /) -> T: ...


@overload
def segmenter_dataclass(
    *,
    eq: bool = False,
    repr: bool = False,
    slots: bool = True,
    frozen: bool = False,
    kw_only: bool = False,
    config: ConfigDict | None = None,
) -> Callable[[T], T]: ...


@dataclass_transform(field_specifiers=(Field,))
def segmenter_dataclass(
    cls: T | None = None,
    /,
    *,
    eq: bool = False,
    repr: bool = False,
    slots: bool = True,
    frozen: bool = False,
    kw_only: bool = False,
    config: ConfigDict | None = None,
) -> T | Callable[[T], T]:
    """Turn *cls* into a pydantic dataclass with the package defaults."""

    def wrap(cls: T) -> T:
        _shadow_inherited_properties(cls)
        new = _pydantic_dataclass(
            cls,
            config=config or MODEL_CONFIG,
            eq=eq,
            repr=repr,
            slots=slots,
            frozen=frozen,
            kw_only=kw_only,
        )
        if new is not cls:
            new.__qualname__ = cls.__qualname__  # not carried over by the slots rewrite
            _rebind_class_cells(cls, new)
            # Property subclasses registered themselves (in __init_subclass__)
            # under the class object that slots replaced.
            registry = sys.modules.get("spice_segmenter.core.registry")
            if registry is not None:
                registry.property_registry.rebind(cls, new)
        return new

    if cls is None:
        return wrap
    return wrap(cls)
