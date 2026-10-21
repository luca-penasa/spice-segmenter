"""
Serialization module for spice_segmenter Properties and Constraints.

Properties and constraints are pydantic dataclasses; this module turns them
into plain, JSON-compatible dicts and back. A node is tagged with its class
name under ``"type"``::

    {
        "observer": "JUICE_JANUS",
        "target": "GANYMEDE",
        "light_time_correction": "NONE",
        "unit": "kilometer",
        "type": "Distance",
    }

Nested nodes (``Constraint.left`` / ``right``, ``Inverted.parent``,
``ComponentSelector.vector``, ...) are fields of type
:data:`~spice_segmenter.support.types.PropertyNode`, which serialize the same
way, so a constraint tree is a nested tagged dict.

:func:`json_schema` describes the format (one definition per class).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.constraints import Constraint
    from ..core.property import Property

TAG = "type"


def get_all_subclasses(cls: type) -> list[type]:
    """
    Recursively get all subclasses of a class.

    Args:
        cls: The base class to get subclasses for

    Returns:
        List of all subclasses (including nested subclasses)
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def node_classes() -> dict[str, type[Property]]:
    """Return the serializable classes (every ``Property`` subclass) by class name.

    A slotted dataclass replaces the class object it was defined as; the
    replaced object can still show up among the subclasses, so classes that
    carry their own pydantic validator win over those that do not.
    """
    from ..core.property import Property

    classes: dict[str, type[Property]] = {}
    for cls in [Property, *get_all_subclasses(Property)]:
        existing = classes.get(cls.__name__)
        if existing is None or (
            "__pydantic_validator__" in cls.__dict__
            and "__pydantic_validator__" not in existing.__dict__
        ):
            classes[cls.__name__] = cls
    return classes


def unstructure_node(obj: Any) -> Any:
    """Serialize a property or constraint to a tagged, JSON-compatible dict.

    Anything that is not a pydantic dataclass is returned unchanged.
    """
    serializer = getattr(type(obj), "__pydantic_serializer__", None)
    if serializer is None:
        return obj
    from .fields import fields

    # Fields computed in __post_init__ (init=False) are not part of the format.
    computed = {f.name for f in fields(type(obj)) if not f.init}
    data = serializer.to_python(obj, mode="json", exclude=computed or None)
    data[TAG] = type(obj).__name__
    return data


def structure_node(data: Any, cls: type | None = None) -> Any:
    """Rebuild a property or constraint from a tagged dict.

    Parameters
    ----------
    data:
        Dict produced by :func:`unstructure_node` (or a JSON/YAML load of it).
        Property / constraint instances are returned unchanged.
    cls:
        Optional expected base class; the result must be an instance of it.
    """
    if not isinstance(data, dict):
        if cls is not None and not isinstance(data, cls):
            raise TypeError(
                f"Expected a dict tagged with {TAG!r} or a {cls.__name__}, got {data!r}"
            )
        return data

    if TAG not in data:
        raise ValueError(
            f"Cannot structure {data}: expected a dict with a {TAG!r} field containing a known class name.",
        )
    type_name = data[TAG]
    classes = node_classes()
    if type_name not in classes:
        raise ValueError(
            f"Cannot structure {data}: unknown {TAG} {type_name!r}. Known: {', '.join(sorted(classes))}",
        )
    target = classes[type_name]
    if cls is not None and not issubclass(target, cls):
        raise TypeError(f"{type_name} is not a {cls.__name__}")

    kwargs = {k: v for k, v in data.items() if k != TAG}
    return target(**kwargs)


class PropertyConverter:
    """Converter with the ``unstructure`` / ``structure`` API of the former cattrs converter."""

    def unstructure(self, obj: Any) -> Any:
        """Serialize *obj* (a property, a constraint, or a list/dict of them)."""
        if isinstance(obj, (list, tuple)):
            return [self.unstructure(o) for o in obj]
        if isinstance(obj, dict) and TAG not in obj:
            return {k: self.unstructure(v) for k, v in obj.items()}
        return unstructure_node(obj)

    def structure(self, data: Any, cls: type | None = None) -> Any:
        """Rebuild an instance of *cls* (or of the class named by ``"type"``) from *data*."""
        return structure_node(data, cls)


def create_property_converter(
    register_spice_types: bool = True,
    register_pint: bool = True,
) -> PropertyConverter:
    """
    Create a converter for spice_segmenter Property and Constraint objects.

    The format is a tagged dict with the class name under ``'type'``; SPICE
    references and pint units/quantities are serialized as strings. The
    arguments are accepted for compatibility and have no effect.

    Returns:
        Converter with ``unstructure(obj)`` and ``structure(data, cls)``

    Example:
        >>> from ..support.serialization import (
        ...     create_property_converter,
        ... )
        >>> from spice_segmenter import (
        ...     PhaseAngle,
        ...     Distance,
        ... )
        >>>
        >>> # Create constraints
        >>> c1 = (
        ...     PhaseAngle(
        ...         "JUICE_JANUS",
        ...         "GANYMEDE",
        ...     )
        ...     > "20 deg"
        ... )
        >>> c2 = (
        ...     Distance(
        ...         "JUICE_JANUS",
        ...         "GANYMEDE",
        ...     )
        ...     < "5e4 km"
        ... )
        >>> cc = c1 & c2
        >>>
        >>> # Create converter and serialize
        >>> converter = create_property_converter()
        >>> data = converter.unstructure(
        ...     cc
        ... )
        >>>
        >>> # Deserialize back
        >>> from spice_segmenter import (
        ...     Constraint,
        ... )
        >>> reconstructed = converter.structure(
        ...     data, Constraint
        ... )
    """
    return PropertyConverter()


def unstructure_constraint(
    constraint: Constraint, converter: PropertyConverter | None = None
) -> dict:
    """
    Convenience function to unstructure (serialize) a Constraint to a dictionary.

    Args:
        constraint: The Constraint object to serialize
        converter: Optional converter. If None, creates a new one.

    Returns:
        Dictionary representation of the constraint
    """
    if converter is None:
        converter = create_property_converter()
    return converter.unstructure(constraint)


def structure_constraint(
    data: dict, converter: PropertyConverter | None = None
) -> Constraint:
    """
    Convenience function to structure (deserialize) a dictionary into a Constraint.

    Args:
        data: Dictionary representation of a constraint
        converter: Optional converter. If None, creates a new one.

    Returns:
        Reconstructed Constraint object
    """
    from ..core.constraints import ConstraintBase

    if converter is None:
        converter = create_property_converter()
    return converter.structure(data, ConstraintBase)


def _link_nodes(schema: Any) -> Any:
    """Point the placeholders of nested-node fields at ``#/$defs/Node``."""
    from .types import NODE_SCHEMA_MARKER

    if isinstance(schema, dict):
        if schema.get(NODE_SCHEMA_MARKER):
            return {"$ref": "#/$defs/Node"}
        return {k: _link_nodes(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_link_nodes(v) for v in schema]
    return schema


def json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of a serialized property or constraint tree.

    Each class gets a definition (its fields plus a constant ``"type"``);
    ``#/$defs/Node`` is any of them, discriminated by ``"type"``. Built from
    the classes currently imported, so plug-in properties are included once
    their module is loaded.
    """
    from pydantic import TypeAdapter

    from ..core.property import Property

    defs: dict[str, Any] = {}
    for name, cls in sorted(node_classes().items()):
        if cls is Property or "__pydantic_validator__" not in cls.__dict__:
            continue
        try:
            schema = TypeAdapter(cls).json_schema(by_alias=False, mode="serialization")
        except Exception:
            continue
        schema = _link_nodes(schema)
        schema.setdefault("properties", {})[TAG] = {"const": name}
        schema["required"] = [TAG, *schema.get("required", [])]
        defs[name] = schema
    defs["Node"] = {
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in defs],
        "discriminator": {"propertyName": TAG},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/Node",
        "$defs": defs,
    }
