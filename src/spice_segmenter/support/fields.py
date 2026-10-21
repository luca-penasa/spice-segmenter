"""Field introspection for the package's pydantic dataclasses.

Replaces ``attrs.fields`` / ``attrs.NOTHING`` / ``attrs.Factory`` checks in the
registry, the DSL and the YAML I/O with one helper that reads a pydantic
dataclass' fields::

    for f in fields(
        Distance
    ):
        (
            f.name,
            f.init_name,
            f.init,
            f.has_default,
            f.has_factory,
        )
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from pydantic_core import PydanticUndefined

#: Sentinel for "no default" (the counterpart of ``attrs.NOTHING``).
MISSING: Any = dataclasses.MISSING


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Description of one field of a pydantic dataclass."""

    name: str
    #: Keyword accepted by ``__init__`` (the alias when there is one, e.g.
    #: ``value`` for the ``_value`` field).
    init_name: str
    init: bool
    kw_only: bool
    #: Static default, or :data:`MISSING`.
    default: Any
    #: Default factory, or ``None``.
    default_factory: Callable[[], Any] | None
    annotation: Any

    @property
    def has_default(self) -> bool:
        """``True`` when the field has a default value or a default factory."""
        return self.default is not MISSING or self.default_factory is not None

    @property
    def has_factory(self) -> bool:
        """``True`` when the default comes from a factory (e.g. the SpiceContext getters)."""
        return self.default_factory is not None

    def get_default(self) -> Any:
        """Return the default, calling the factory if needed; :data:`MISSING` if none."""
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


def is_model(cls: type) -> bool:
    """``True`` when *cls* is a pydantic dataclass."""
    return (
        isinstance(cls, type)
        and hasattr(cls, "__pydantic_fields__")
        and dataclasses.is_dataclass(cls)
    )


def fields(cls: type) -> tuple[FieldSpec, ...]:
    """Return the fields of the pydantic dataclass *cls*, in definition order.

    Raises
    ------
    TypeError
        If *cls* is not a pydantic dataclass.
    """
    if not is_model(cls):
        raise TypeError(f"{cls!r} is not a pydantic dataclass")

    infos = cls.__pydantic_fields__
    out = []
    for dc_field in dataclasses.fields(cls):
        info = infos.get(dc_field.name)
        if info is None:
            continue
        default = MISSING if info.default is PydanticUndefined else info.default
        init = dc_field.init if info.init is None else info.init
        kw_only = bool(dc_field.kw_only) if info.kw_only is None else info.kw_only
        out.append(
            FieldSpec(
                name=dc_field.name,
                init_name=info.alias or dc_field.name,
                init=init,
                kw_only=kw_only,
                default=default,
                default_factory=info.default_factory,
                annotation=info.annotation,
            ),
        )
    return tuple(out)


def evolve(obj: Any, **changes: Any) -> Any:
    """Return a copy of *obj* with *changes* applied (the ``attrs.evolve`` counterpart).

    Like :func:`dataclasses.replace`, but aware of pydantic ``init=False``
    fields (recomputed by ``__post_init__``, never passed back) and aliases.
    *changes* are keyed by field name; the copy is validated again.
    """
    kwargs = {}
    for f in fields(type(obj)):
        if not f.init:
            if f.name in changes:
                raise ValueError(
                    f"field {f.name!r} is init=False and cannot be changed"
                )
            continue
        kwargs[f.name] = (
            changes.pop(f.name) if f.name in changes else getattr(obj, f.name)
        )
    kwargs.update(changes)  # unknown names: let the constructor reject them
    return type(obj)(**kwargs)


def field_names(obj_or_cls: Any) -> tuple[str, ...]:
    """Names of the fields of a pydantic dataclass or of one of its instances."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return tuple(f.name for f in fields(cls))
