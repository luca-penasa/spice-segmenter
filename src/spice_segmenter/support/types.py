"""Annotated pydantic field types: conversion, serialization and JSON schema.

Each type replaces one kind of attrs ``converter=``: a ``BeforeValidator``
converts the input (as the converter did), a ``PlainSerializer`` gives the
wire form (the one the former cattrs converter produced) and ``WithJsonSchema``
describes it::

    @segmenter_dataclass
    class PhaseAngle(
        TargetedProperty
    ):
        unit: UnitField = Field(
            default="rad",
            kw_only=True,
        )
        third_body: SpiceRefField = Field(
            default="SUN"
        )

:func:`converted` builds a type for any other conversion function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import pint
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


def converted(
    fn: Callable[[Any], Any],
    *,
    serializer: Callable[[Any], Any] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> Any:
    """Annotated type that runs *fn* on the input, like an attrs converter.

    Parameters
    ----------
    fn:
        Conversion applied to every value (explicit or default).
    serializer:
        Function giving the serialized (JSON) form of the value.
    json_schema:
        JSON schema of the serialized form.
    """
    # Wrapped so that pydantic sees a one-argument function whatever the
    # signature of fn (builtins, numpy functions with *args, ...).
    metadata: list[Any] = [BeforeValidator(lambda value: fn(value))]
    if serializer is not None:
        metadata.append(PlainSerializer(lambda value: serializer(value)))
    if json_schema is not None:
        metadata.append(WithJsonSchema(json_schema))
    return Annotated[(Any, *metadata)]


# ---------------------------------------------------------------------------
# pint
# ---------------------------------------------------------------------------


def to_pint_unit(x: pint.Unit | str | tuple | list | None) -> pint.Unit | tuple:
    """Convert to ``pint.Unit``: strings are parsed, tuples/lists converted
    element-wise, existing ``pint.Unit`` pass through.
    """
    if isinstance(x, (list, tuple)):
        return tuple(to_pint_unit(u) for u in x)
    if isinstance(x, str):
        return pint.Unit(x)
    if isinstance(x, pint.Unit):
        return x
    if x is None:
        raise TypeError("unit cannot be None; provide a unit string or pint.Unit")
    return pint.Unit(str(x))  # fallback for pint-compatible objects


def _unit_to_str(u: pint.Unit | tuple) -> str | list[str]:
    if isinstance(u, (list, tuple)):
        return [str(x) for x in u]
    return str(u)


def _str_or_none(v: Any) -> str | None:
    return None if v is None else str(v)


#: pint unit (or tuple of units for vector properties); serialized as text, e.g. ``"kilometer"``.
UnitField = converted(
    to_pint_unit,
    serializer=_unit_to_str,
    json_schema={
        "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
        "description": "pint unit, e.g. 'km', 'deg', 'rad/s'",
    },
)

#: Like :data:`UnitField`, ``None`` allowed.
OptionalUnitField = converted(
    lambda x: None if x is None else to_pint_unit(x),
    serializer=lambda u: None if u is None else _unit_to_str(u),
    json_schema={
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "pint unit, or null",
    },
)

#: pint quantity; serialized as text, e.g. ``"20 degree"``.
QuantityField = converted(
    lambda x: x if isinstance(x, pint.Quantity) else pint.Quantity(x),
    serializer=_str_or_none,
    json_schema={"type": "string", "description": "quantity with unit, e.g. '20 deg'"},
)


# ---------------------------------------------------------------------------
# SPICE references (planetary_coverage)
# ---------------------------------------------------------------------------

_SPICE_SCHEMA = {"type": "string", "description": "SPICE name or NAIF id"}


def _spice(cls_name: str, description: str) -> Any:
    def convert(x: Any) -> Any:
        import planetary_coverage.spice as pcs

        return getattr(pcs, cls_name)(x)

    return converted(
        convert,
        serializer=_str_or_none,
        json_schema={**_SPICE_SCHEMA, "description": description},
    )


def _as_spice_ref(x: Any) -> Any:
    from .spice_utilities import as_spice_ref

    return as_spice_ref(x)


#: Any SPICE reference (body, spacecraft, instrument, ...) through ``SpiceRef``.
SpiceRefField = _spice(
    "SpiceRef", "SPICE name or NAIF id (body, spacecraft, instrument)"
)
#: SPICE body.
SpiceBodyField = _spice("SpiceBody", "SPICE body name or NAIF id")
#: SPICE instrument.
SpiceInstrumentField = _spice("SpiceInstrument", "SPICE instrument name or NAIF id")
#: SPICE observer (spacecraft or instrument).
SpiceObserverField = _spice(
    "SpiceObserver", "SPICE observer (spacecraft or instrument) name or NAIF id"
)
#: SPICE frame.
SpiceFrameField = _spice("SpiceFrame", "SPICE frame name")
#: SPICE reference, existing references passed through unchanged (``as_spice_ref``).
AsSpiceRefField = converted(
    _as_spice_ref,
    serializer=_str_or_none,
    json_schema={
        **_SPICE_SCHEMA,
        "description": "SPICE name or NAIF id (body, spacecraft, instrument)",
    },
)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _to_seconds(x: Any) -> float | None:
    import pandas as pd

    return x if isinstance(x, float | None) else pd.Timedelta(x).total_seconds()


#: Duration in seconds; strings and timedeltas are converted (``"10 min"`` -> 600.0).
SecondsField = converted(
    _to_seconds,
    json_schema={"type": ["number", "null"], "description": "duration in seconds"},
)


# ---------------------------------------------------------------------------
# Nested properties / constraints
# ---------------------------------------------------------------------------


NODE_SCHEMA_MARKER = "x-spice-segmenter-node"


def _load_node(v: Any) -> Any:
    if isinstance(v, dict) and "type" in v:
        from .serialization import structure_node

        return structure_node(v)
    return v


def _dump_node(v: Any) -> Any:
    from .serialization import unstructure_node

    return unstructure_node(v)


#: A nested :class:`~spice_segmenter.core.property.Property` or constraint;
#: serialized as a tagged dict ``{"type": "<ClassName>", ...fields}``.
PropertyNode = Annotated[
    Any,
    BeforeValidator(_load_node),
    PlainSerializer(_dump_node),
    # Replaced by a reference to the whole node union in serialization.json_schema().
    WithJsonSchema({"type": "object", NODE_SCHEMA_MARKER: True}),
]
