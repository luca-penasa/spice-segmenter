"""YAML serialization and deserialization for constraints and property instances.

Constraint file format
----------------------
::

    context:
      observer: JUICE_JANUS
      target: GANYMEDE
      light_time_correction: NONE

    # Optional property-specific overrides (params outside context)
    properties:
      phase_angle:
        third_body: SUN
      shine_properties:
        reflector: JUPITER

    expression: "(distance > '2 km' | phase_angle < '2 deg') & fov_visibility"

Property-list file format
-------------------------
::

    context:
      observer: JUICE_JANUS
      target: GANYMEDE
      light_time_correction: NONE

    properties:
      - type: distance

      - type: phase_angle
        third_body: SUN        # overrides class default

      - type: distance
        target: TITAN          # shadows context.target for this entry only

      - type: shine_properties
        reflector: JUPITER     # required field not provided by context

Public API — constraints
------------------------
``loads(yaml_str)`` / ``load(path)``  — deserialize → :class:`~spice_segmenter.core.constraints.ConstraintBase`
``dumps(constraint)`` / ``dump(constraint, path)``  — serialize → YAML string / file

Public API — property lists
----------------------------
``loads_properties(yaml_str)`` / ``load_properties(path)``  — deserialize → ``list[Property]``
``dumps_properties(properties)`` / ``dump_properties(properties, path)``  — serialize → YAML string / file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import yaml

from spice_segmenter.core.constraints import ConstraintBase
from spice_segmenter.core.property import Property
from spice_segmenter.io.dsl import (
    _CONTEXT_FIELDS,
    _make_property,
    _spice_ref_to_str,
    constraint_to_context,
    constraint_to_expression,
    parse,
)


# ---------------------------------------------------------------------------
# Constraint deserialization
# ---------------------------------------------------------------------------

def loads(yaml_str: str) -> ConstraintBase:
    """Parse a YAML string into a :class:`~spice_segmenter.core.constraints.ConstraintBase`.

    Parameters
    ----------
    yaml_str:
        Full YAML document string.

    Returns
    -------
    ConstraintBase
        The reconstructed constraint.

    Examples
    --------
    >>> c = loads('''
    ... context:
    ...   observer: JUICE_JANUS
    ...   target: GANYMEDE
    ... expression: "distance > '1e4 km'"
    ... ''')
    """
    data = yaml.safe_load(yaml_str)
    return _constraint_from_dict(data)


def load(path: str | Path) -> ConstraintBase:
    """Load a constraint from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    ConstraintBase
        The reconstructed constraint.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return _constraint_from_dict(data)


def _constraint_from_dict(data: dict[str, Any]) -> ConstraintBase:
    if not isinstance(data, dict):
        raise ValueError("YAML document must be a mapping at the top level.")

    expression = data.get("expression")
    if not expression:
        raise ValueError("YAML document must contain an 'expression' key.")

    context: dict[str, Any] = data.get("context") or {}
    overrides: dict[str, dict[str, Any]] = data.get("properties") or {}

    return parse(expression, context=context, overrides=overrides)


# ---------------------------------------------------------------------------
# Constraint serialization
# ---------------------------------------------------------------------------

def dumps(constraint: ConstraintBase) -> str:
    """Serialize a constraint to a YAML string.

    Parameters
    ----------
    constraint:
        The constraint to serialize.

    Returns
    -------
    str
        A YAML string that :func:`loads` can round-trip back.
    """
    data = _constraint_to_dict(constraint)
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def dump(constraint: ConstraintBase, path: str | Path) -> None:
    """Serialize a constraint to a YAML file.

    Parameters
    ----------
    constraint:
        The constraint to serialize.
    path:
        Destination file path.  Parent directories must exist.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(dumps(constraint))


def _constraint_to_dict(constraint: ConstraintBase) -> dict[str, Any]:
    expression = constraint_to_expression(constraint)
    context, overrides = constraint_to_context(constraint)

    data: dict[str, Any] = {}
    if context:
        data["context"] = context
    if overrides:
        data["properties"] = overrides
    data["expression"] = expression
    return data


# ---------------------------------------------------------------------------
# Property-list: deserialization helpers
# ---------------------------------------------------------------------------

def _entry_to_property(entry: dict[str, Any], shared_context: dict[str, Any]) -> Property:
    """Instantiate a single property from a list-entry dict + shared context."""
    entry = dict(entry)

    prop_type = entry.pop("type", None)
    if not prop_type:
        raise ValueError(f"Property entry is missing a 'type' key: {entry!r}")

    merged: dict[str, Any] = {**shared_context, **entry}
    context_kwargs = {k: v for k, v in merged.items() if k in _CONTEXT_FIELDS}
    extra_kwargs = {k: v for k, v in merged.items() if k not in _CONTEXT_FIELDS}

    return _make_property(prop_type, context_kwargs, extra_kwargs)


# ---------------------------------------------------------------------------
# Property-list: deserialization public API
# ---------------------------------------------------------------------------

def loads_properties(yaml_str: str) -> list[Property]:
    """Deserialize a YAML string into a list of :class:`~spice_segmenter.core.property.Property` instances.

    Parameters
    ----------
    yaml_str:
        Full YAML document string using the property-list format.

    Returns
    -------
    list[Property]
        Instantiated property objects in the order they appear in the file.

    Examples
    --------
    >>> props = loads_properties('''
    ... context:
    ...   observer: JUICE_JANUS
    ...   target: GANYMEDE
    ... properties:
    ...   - type: distance
    ...   - type: phase_angle
    ... ''')
    """
    data = yaml.safe_load(yaml_str)
    return _properties_from_dict(data)


def load_properties(path: str | Path) -> list[Property]:
    """Load a list of property instances from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML file using the property-list format.

    Returns
    -------
    list[Property]
        Instantiated property objects in the order they appear in the file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return _properties_from_dict(data)


def _properties_from_dict(data: dict[str, Any]) -> list[Property]:
    if not isinstance(data, dict):
        raise ValueError("YAML document must be a mapping at the top level.")

    entries = data.get("properties")
    if not entries:
        raise ValueError("YAML document must contain a 'properties' key with a list of entries.")
    if not isinstance(entries, list):
        raise ValueError("'properties' must be a list of entries (each with a 'type' key).")
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(
                f"Entry {i} in 'properties' must be a mapping with at least a 'type' key, "
                f"got {type(e).__name__}: {e!r}"
            )

    shared_context: dict[str, Any] = data.get("context") or {}
    return [_entry_to_property(e, shared_context) for e in entries]


# ---------------------------------------------------------------------------
# Property-list: serialization helpers
# ---------------------------------------------------------------------------

def _get_field_default(cls: type, field: attrs.Attribute) -> Any:
    """Return the default value of an attrs field, evaluating factories."""
    if field.default is attrs.NOTHING:
        return attrs.NOTHING
    if isinstance(field.default, attrs.Factory):  # type: ignore[arg-type]
        if field.default.takes_self:  # type: ignore[union-attr]
            return attrs.NOTHING
        try:
            return field.default.factory()  # type: ignore[union-attr]
        except Exception:
            return attrs.NOTHING
    return field.default


def _property_to_entry(prop: Property, shared_context: dict[str, Any]) -> dict[str, Any]:
    """Serialize a single property to a YAML entry dict."""
    entry: dict[str, Any] = {"type": prop._name}

    try:
        fields = attrs.fields(type(prop))
    except attrs.exceptions.NotAnAttrsClassError:
        return entry

    for f in fields:
        if not f.init or f.name.startswith("_"):
            continue

        raw_val = getattr(prop, f.name)
        str_val = _spice_ref_to_str(raw_val)

        if f.name in _CONTEXT_FIELDS:
            if shared_context.get(f.name) != str_val:
                entry[f.name] = str_val
        else:
            class_default = _get_field_default(type(prop), f)
            default_str = (
                _spice_ref_to_str(class_default)
                if class_default is not attrs.NOTHING
                else attrs.NOTHING
            )
            if default_str is attrs.NOTHING or default_str != str_val:
                entry[f.name] = str_val

    return entry


def _infer_shared_context(properties: list[Property]) -> dict[str, Any]:
    """Promote context fields to shared level when all properties agree."""
    field_values: dict[str, list[str]] = {k: [] for k in _CONTEXT_FIELDS}

    for prop in properties:
        for field_name in _CONTEXT_FIELDS:
            if hasattr(prop, field_name):
                field_values[field_name].append(_spice_ref_to_str(getattr(prop, field_name)))

    shared: dict[str, Any] = {}
    for field_name, values in field_values.items():
        if values and len(set(values)) == 1:
            shared[field_name] = values[0]

    return shared


# ---------------------------------------------------------------------------
# Property-list: serialization public API
# ---------------------------------------------------------------------------

def dumps_properties(properties: list[Property]) -> str:
    """Serialize a list of property instances to a YAML string.

    Parameters
    ----------
    properties:
        List of :class:`~spice_segmenter.core.property.Property` instances.

    Returns
    -------
    str
        A YAML string that :func:`loads_properties` can round-trip back.
    """
    data = _properties_to_dict(properties)
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def dump_properties(properties: list[Property], path: str | Path) -> None:
    """Serialize a list of property instances to a YAML file.

    Parameters
    ----------
    properties:
        List of :class:`~spice_segmenter.core.property.Property` instances.
    path:
        Destination file path.  Parent directories must exist.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(dumps_properties(properties))


def _properties_to_dict(properties: list[Property]) -> dict[str, Any]:
    shared_context = _infer_shared_context(properties)
    entries = [_property_to_entry(p, shared_context) for p in properties]
    data: dict[str, Any] = {}
    if shared_context:
        data["context"] = shared_context
    data["properties"] = entries
    return data
