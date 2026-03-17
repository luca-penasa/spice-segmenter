"""Constraint mini-language: parse and serialize.

Parse
-----
The DSL is a Python-expression subset evaluated with :func:`parse`::

    from spice_segmenter.io.dsl import parse

    c = parse(
        "(distance > '2 km' | phase_angle < '2 deg') & fov_visibility",
        context={"observer": "JUICE_JANUS", "target": "GANYMEDE"},
    )

Grammar summary
~~~~~~~~~~~~~~~
Two equivalent syntaxes are supported:

**Boolean operators** ``or`` / ``and`` / ``not`` — *recommended*; natural Python
precedence means comparisons bind tighter so no extra parens are needed::

    (distance > '2 km' or phase_angle < '2 deg') and fov_visibility

**Bitwise operators** ``|`` / ``&`` / ``~`` — bind *tighter* than comparisons
in Python, so each comparison must be individually parenthesised::

    ((distance > '2 km') | (phase_angle < '2 deg')) & fov_visibility

The serializer always emits parenthesised bitwise form so the output is
unambiguous and explicit.

Other constructs:

- ``prop``       — bare boolean shorthand: equivalent to ``prop == True``
- ``prop[0]``    — vector component by index
- ``prop.sub``   — vector component by named sub-property
- ``Enum.MEMBER``— enum value (e.g. ``OccultationTypes.FULL``)

Serialize
---------
:func:`constraint_to_expression` converts a :class:`~spice_segmenter.core.constraints.Constraint`
back to a DSL string.  :func:`constraint_to_context` extracts the shared
observer/target context and per-property overrides so they can be stored
alongside the expression string.
"""

from __future__ import annotations

import ast
from typing import Any

import attrs
import pint

from spice_segmenter.core.constraints import Constraint, ConstraintBase
from spice_segmenter.core.registry import property_registry
from spice_segmenter.ops.constant_values import BoolConstant, Constant, ScalarConstant
from spice_segmenter.ops.constraint_operations import Inverted, WrappedConstraint
from spice_segmenter.properties.component_selector import ComponentSelector
from spice_segmenter.support.context import SpiceContext

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fields provided by SpiceContext — never go into property overrides.
_CONTEXT_FIELDS = {"observer", "target", "light_time_correction"}

# Known enum namespaces available in DSL expressions.
_ENUM_NAMESPACES: dict[str, Any] = {}


def _enum_namespaces() -> dict[str, Any]:
    """Lazily build the enum namespace dict."""
    if not _ENUM_NAMESPACES:
        from spice_segmenter.properties.occultation_types import OccultationTypes
        from spice_segmenter.properties.observation_properties import MinMaxConditionTypes

        _ENUM_NAMESPACES["OccultationTypes"] = OccultationTypes
        _ENUM_NAMESPACES["MinMaxConditionTypes"] = MinMaxConditionTypes
    return _ENUM_NAMESPACES


# Mapping from ast comparison operator types to constraint operator strings.
_AST_OP: dict[type, str] = {
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
    ast.Eq: "=",
    ast.NotEq: "!=",
}

# Mapping from constraint operator string → DSL string (for serialization).
_OP_TO_DSL: dict[str, str] = {
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "=": "==",
    "&": "&",
    "|": "|",
}


# ---------------------------------------------------------------------------
# Property instantiation helpers
# ---------------------------------------------------------------------------

def _make_property(name: str, context: dict[str, Any], overrides: dict[str, Any]):
    """Instantiate a property from the registry inside a SpiceContext.

    Parameters
    ----------
    name:
        Registry name (``_name`` class attribute).
    context:
        Keyword arguments forwarded to :class:`SpiceContext` for fields that
        use context factories (``observer``, ``target``, ``light_time_correction``
        with ``Factory`` defaults).  For properties whose context-named fields
        have explicit defaults or no default at all (e.g. :class:`Occultation`),
        those values are passed directly to the constructor instead.
    overrides:
        Extra keyword arguments passed directly to the property constructor.
    """
    if name not in property_registry:
        available = ", ".join(sorted(property_registry))
        raise KeyError(
            f"Unknown property {name!r} in expression. "
            f"Registered properties: {available}"
        )
    cls = property_registry[name]

    # Determine which context fields this class actually reads from SpiceContext
    # (i.e. those whose default is an attrs Factory).  Context fields without a
    # Factory default must be supplied directly to the constructor.
    try:
        cls_fields = {f.name: f for f in attrs.fields(cls)}
    except attrs.exceptions.NotAnAttrsClassError:
        cls_fields = {}

    factory_fields = {
        fname for fname, f in cls_fields.items() if isinstance(f.default, attrs.Factory)
    }
    # Context kwargs whose class field uses a factory → delegated to SpiceContext.
    spice_ctx_kwargs = {k: v for k, v in context.items() if k in factory_fields}
    # Context kwargs whose class field does NOT use a factory → pass to constructor.
    direct_ctx_kwargs = {k: v for k, v in context.items() if k not in factory_fields}

    # Merge: direct context kwargs + overrides (overrides win for conflicts).
    constructor_kwargs = {**direct_ctx_kwargs, **overrides}

    with SpiceContext(**spice_ctx_kwargs):
        return cls(**constructor_kwargs)


# ---------------------------------------------------------------------------
# AST walker — parse expression into a Constraint
# ---------------------------------------------------------------------------

def _eval_value(node: ast.expr) -> Constant:
    """Evaluate a value node into a :class:`Constant`."""
    if isinstance(node, ast.Constant):
        return Constant.from_value(node.value)

    # Enum member: OccultationTypes.FULL
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        ns = _enum_namespaces()
        enum_cls = ns.get(node.value.id)
        if enum_cls is not None:
            member = getattr(enum_cls, node.attr)
            return Constant.from_value(member)

    raise ValueError(f"Unsupported value node: {ast.dump(node)}")


def _eval_prop_ref(node: ast.expr, prop_map: dict[str, Any]):
    """Resolve a property reference node to a Property instance."""
    # Simple name: distance, fov_visibility
    if isinstance(node, ast.Name):
        name = node.id
        if name not in prop_map:
            raise KeyError(f"Unknown property {name!r}. Available: {', '.join(sorted(prop_map))}")
        return prop_map[name]

    # Indexed component: shine_properties[0]
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise ValueError(f"Subscript target must be a name, got {ast.dump(node.value)}")
        parent = prop_map[node.value.id]
        if isinstance(node.slice, ast.Constant):
            idx = int(node.slice.value)
        else:
            raise ValueError("Subscript index must be an integer constant")
        return ComponentSelector(parent, idx)

    # Named sub-property: shine_properties.reflector_elevation
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        parent_name = node.value.id
        if parent_name in prop_map:
            parent = prop_map[parent_name]
            attr = node.attr
            if not hasattr(parent, attr):
                raise AttributeError(
                    f"Property {parent_name!r} has no sub-property {attr!r}"
                )
            return getattr(parent, attr)

    raise ValueError(f"Unsupported property reference: {ast.dump(node)}")


def _is_prop_ref(node: ast.expr, prop_map: dict[str, Any]) -> bool:
    """Return True if *node* looks like a property reference (name / subscript / attribute)."""
    if isinstance(node, ast.Name) and node.id in prop_map:
        return True
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id in prop_map
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id not in _enum_namespaces() and node.value.id in prop_map
    return False


def _eval_expr(node: ast.expr, prop_map: dict[str, Any]) -> ConstraintBase:
    """Recursively evaluate an AST expression node into a Constraint.

    Supports two equivalent syntaxes:

    * Bitwise operators ``|`` / ``&`` / ``~``:
      Because Python's bitwise operators bind *tighter* than comparisons,
      comparisons must be individually parenthesised::

          (distance > '2 km') | (phase_angle < '2 deg')

    * Boolean operators ``or`` / ``and`` / ``not``:
      Python's boolean operators bind *looser* than comparisons, so
      parentheses around individual comparisons are not required::

          distance > '2 km' or phase_angle < '2 deg'
    """

    # ── Boolean OR: a or b  (lower prec than comparisons — natural syntax)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        result = _eval_expr(node.values[0], prop_map)
        for value in node.values[1:]:
            result = result | _eval_expr(value, prop_map)
        return result

    # ── Boolean AND: a and b
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        result = _eval_expr(node.values[0], prop_map)
        for value in node.values[1:]:
            result = result & _eval_expr(value, prop_map)
        return result

    # ── Boolean NOT: not a
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ~_eval_expr(node.operand, prop_map)

    # ── Bitwise OR: a | b  (needs parens around comparisons)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _eval_expr(node.left, prop_map) | _eval_expr(node.right, prop_map)

    # ── Bitwise AND: a & b
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        return _eval_expr(node.left, prop_map) & _eval_expr(node.right, prop_map)

    # ── Bitwise NOT / Invert: ~a
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return ~_eval_expr(node.operand, prop_map)

    # ── Comparison: distance > '2 km'
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left_prop = _eval_prop_ref(node.left, prop_map)
        op_type = type(node.ops[0])
        if op_type not in _AST_OP:
            raise ValueError(f"Unsupported comparison operator: {ast.dump(node.ops[0])}")
        op_str = _AST_OP[op_type]
        right_val = _eval_value(node.comparators[0])
        return Constraint(left_prop, right_val, op_str)

    # ── Bare property reference: fov_visibility  →  fov_visibility == True
    if _is_prop_ref(node, prop_map):
        prop = _eval_prop_ref(node, prop_map)
        return Constraint(prop, BoolConstant(True), "=")

    raise ValueError(
        f"Unsupported expression node: {ast.dump(node)}\n"
        "The DSL supports:\n"
        "  Boolean ops (natural precedence):  a or b,  a and b,  not a\n"
        "  Bitwise ops (parens around cmps):  (a > x) | (b < y),  (a) & (b),  ~a\n"
        "  Comparisons:  prop > 'value unit',  prop == True\n"
        "  Bare boolean: fov_visibility  (shorthand for fov_visibility == True)"
    )


# ---------------------------------------------------------------------------
# Public parse API
# ---------------------------------------------------------------------------

def parse(
    expression: str,
    context: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ConstraintBase:
    """Parse a DSL expression string into a :class:`~spice_segmenter.core.constraints.Constraint`.

    Parameters
    ----------
    expression:
        DSL string, e.g.
        ``"(distance > '2 km' | phase_angle < '2 deg') & fov_visibility"``.
    context:
        Keyword arguments for :class:`~spice_segmenter.support.context.SpiceContext`
        (``observer``, ``target``, ``light_time_correction``).
    overrides:
        Per-property keyword arguments, keyed by property name.
        E.g. ``{"phase_angle": {"third_body": "SUN"}, "shine_properties": {"reflector": "JUPITER"}}``.

    Returns
    -------
    ConstraintBase
        The parsed, fully instantiated constraint.

    Examples
    --------
    >>> c = parse(
    ...     "(distance > '2 km' | phase_angle < '2 deg') & fov_visibility",
    ...     context={"observer": "JUICE_JANUS", "target": "GANYMEDE"},
    ... )
    """
    ctx = context or {}
    ovr = overrides or {}

    class _LazyPropMap:
        def __init__(self):
            self._cache: dict[str, Any] = {}

        def __contains__(self, name: str) -> bool:
            return name in property_registry

        def __getitem__(self, name: str):
            if name not in self._cache:
                self._cache[name] = _make_property(name, ctx, ovr.get(name, {}))
            return self._cache[name]

        def __iter__(self):
            return iter(property_registry)

    prop_map = _LazyPropMap()

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid DSL expression: {exc}") from exc

    return _eval_expr(tree.body, prop_map)


# ---------------------------------------------------------------------------
# Serialize: Constraint → DSL expression string
# ---------------------------------------------------------------------------

def _serialize_value(constant: Constant) -> str:
    """Convert a Constant back to a DSL value token."""
    if isinstance(constant, BoolConstant):
        return str(constant.value)
    if isinstance(constant, ScalarConstant):
        q = constant._value  # pint.Quantity
        if str(q.units) == "dimensionless":
            from enum import Enum
            mag = q.magnitude
            if isinstance(mag, Enum):
                return f"{type(mag).__name__}.{mag.name}"
            return repr(mag)
        return f"'{q:~}'"
    return repr(constant)


def _serialize_prop_ref(prop) -> str:
    """Convert a Property instance back to a DSL reference token."""
    if isinstance(prop, ComponentSelector):
        parent_name = prop.vector.name
        sub_name = prop._name
        if sub_name == "component_selector":
            return f"{parent_name}[{prop.component}]"
        return f"{parent_name}.{sub_name}"
    return prop.name


def _is_simple_comparison(c: ConstraintBase) -> bool:
    """True for leaf comparisons that need extra parens when inside | or &."""
    if not isinstance(c, Constraint) or isinstance(c, (Inverted, WrappedConstraint)):
        return False
    if c.operator in ("&", "|"):
        return False
    if c.operator == "=" and isinstance(c.right, BoolConstant) and c.right.value is True:
        return False
    return True


def constraint_to_expression(constraint: ConstraintBase) -> str:
    """Serialize a :class:`~spice_segmenter.core.constraints.Constraint` to a DSL string.

    Parameters
    ----------
    constraint:
        The constraint tree to serialize.

    Returns
    -------
    str
        A DSL expression string that :func:`parse` can round-trip back.
    """
    if isinstance(constraint, Inverted):
        inner = constraint_to_expression(constraint.parent)
        return f"~({inner})"
    if isinstance(constraint, WrappedConstraint):
        return constraint_to_expression(constraint.parent)

    if isinstance(constraint, Constraint):
        op = constraint.operator
        left = constraint.left
        right = constraint.right

        if op in ("&", "|"):
            left_str = constraint_to_expression(left)  # type: ignore[arg-type]
            right_str = constraint_to_expression(right)  # type: ignore[arg-type]
            if _is_simple_comparison(left):  # type: ignore[arg-type]
                left_str = f"({left_str})"
            if _is_simple_comparison(right):  # type: ignore[arg-type]
                right_str = f"({right_str})"
            return f"({left_str} {_OP_TO_DSL[op]} {right_str})"

        left_str = _serialize_prop_ref(left)
        op_dsl = _OP_TO_DSL.get(op, op)

        if isinstance(right, BoolConstant) and right.value is True and op == "=":
            return left_str

        right_str = _serialize_value(right)  # type: ignore[arg-type]
        return f"{left_str} {op_dsl} {right_str}"

    raise TypeError(f"Cannot serialize constraint of type {type(constraint).__name__}")


# ---------------------------------------------------------------------------
# Serialize: Constraint → context dict + overrides dict
# ---------------------------------------------------------------------------

def _walk_leaf_properties(node) -> list:
    """Recursively collect all leaf Property instances (non-ConstraintBase)."""
    from spice_segmenter.core.property import Property

    if isinstance(node, WrappedConstraint):
        return _walk_leaf_properties(node.parent)
    if isinstance(node, ConstraintBase):
        return _walk_leaf_properties(node.left) + _walk_leaf_properties(node.right)  # type: ignore[arg-type]
    if isinstance(node, ComponentSelector):
        return _walk_leaf_properties(node.vector)
    if isinstance(node, Constant):
        return []
    if isinstance(node, Property):
        return [node]
    return []


def _spice_ref_to_str(val: Any) -> Any:
    """Convert SpiceRef/SpiceBody etc. to their string name where possible."""
    try:
        return val.name
    except AttributeError:
        return val


def constraint_to_context(
    constraint: ConstraintBase,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Extract the shared context and per-property overrides from a constraint tree.

    Returns
    -------
    context : dict
        Keys ``observer``, ``target``, ``light_time_correction`` (as strings).
    overrides : dict
        Keyed by property ``_name``; values are dicts of extra constructor kwargs.
        Only non-default, non-context fields are included.

    Raises
    ------
    ValueError
        If leaf properties have conflicting ``observer``/``target`` values.
    """
    leaves = _walk_leaf_properties(constraint)

    context: dict[str, Any] = {}
    overrides: dict[str, dict[str, Any]] = {}

    for prop in leaves:
        prop_name = _serialize_prop_ref(prop)

        for field_name in ("observer", "target", "light_time_correction"):
            if not hasattr(prop, field_name):
                continue
            val = _spice_ref_to_str(getattr(prop, field_name))
            if field_name in context and context[field_name] != val:
                raise ValueError(
                    f"Constraint contains conflicting {field_name!r} values: "
                    f"{context[field_name]!r} vs {val!r}. "
                    "Cannot be represented as a single context block."
                )
            context[field_name] = val

        try:
            fields = attrs.fields(type(prop))
        except attrs.exceptions.NotAnAttrsClassError:
            continue

        prop_overrides: dict[str, Any] = {}
        for f in fields:
            if not f.init or f.name.startswith("_") or f.name in _CONTEXT_FIELDS:
                continue
            val = _spice_ref_to_str(getattr(prop, f.name))
            prop_overrides[f.name] = val

        if prop_overrides:
            overrides[prop_name] = prop_overrides

    return context, overrides
