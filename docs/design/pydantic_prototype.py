"""Prototype: spice_segmenter's core (Property, Constraint) on pydantic v2, checking the risky points.

Findings: docs/design/pydantic_migration.md. Run from the repository root (needs the JUICE kernels through
quick_spice_manager): ``uv run python docs/design/pydantic_prototype.py``.

1. positional construction + operator overloading (`prop > "20 deg"` -> Constraint, `prop == x` -> Constraint)
2. SPICE / pint field types: validate from str, serialize to str, JSON schema
3. registry-driven tagged union ("type" = class name) for the recursive constraint tree, same wire format as cattrs
4. cost of construction / evaluation vs attrs
"""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any, ClassVar, Union

os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "")
from loguru import logger

logger.remove()

import numpy as np
import pint
import spiceypy as sp
from planetary_coverage.spice import SpiceBody, SpiceRef
from pydantic import (BeforeValidator, ConfigDict, Discriminator, Field, PlainSerializer, Tag, TypeAdapter,
                      WithJsonSchema)
from pydantic.dataclasses import dataclass as pdataclass

from quick_spice_manager import QuickSpiceManager

QuickSpiceManager().load_kernels()

ureg = pint.get_application_registry()

# --- 2. external types as annotated pydantic types ---------------------------------------------------------------
SpiceRefT = Annotated[
    Any,
    BeforeValidator(lambda v: v if isinstance(v, SpiceRef.__mro__[0]) or hasattr(v, "id") else SpiceRef(str(v))),
    PlainSerializer(lambda v: str(v), return_type=str),
    WithJsonSchema({"type": "string", "description": "SPICE name or NAIF id (body, spacecraft, instrument)"}),
]
SpiceBodyT = Annotated[
    Any,
    BeforeValidator(lambda v: v if isinstance(v, SpiceBody) else SpiceBody(str(v))),
    PlainSerializer(lambda v: str(v), return_type=str),
    WithJsonSchema({"type": "string", "description": "SPICE body name or NAIF id"}),
]
UnitT = Annotated[
    Any,
    BeforeValidator(lambda v: v if isinstance(v, pint.Unit) else ureg.Unit(str(v))),
    PlainSerializer(lambda v: f"{v:~}" if str(v) else "", return_type=str),
    WithJsonSchema({"type": "string", "description": "pint unit, e.g. 'km', 'deg', 'rad/s'"}),
]
QuantityT = Annotated[
    Any,
    BeforeValidator(lambda v: v if isinstance(v, pint.Quantity) else ureg.Quantity(str(v))),
    PlainSerializer(lambda v: f"{v:~}", return_type=str),
    WithJsonSchema({"type": "string", "description": "quantity with unit, e.g. '20 deg'"}),
]

REGISTRY: dict[str, type] = {}
CFG = ConfigDict(arbitrary_types_allowed=True, validate_default=True)


# --- 1. Property base: pydantic dataclass, positional args, eq=False keeps our __eq__ -------------------------------
@pdataclass(config=CFG, eq=False, repr=False)
class Property:
    _name: ClassVar[str] = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if getattr(cls, "_name", ""):
            REGISTRY[cls.__name__] = cls

    def _cmp(self, op, other):
        right = other if isinstance(other, (Property, Constraint)) else ScalarConstant(value=other)
        return Constraint(left=self, operator=op, right=right)

    def __gt__(self, other):
        return self._cmp(">", other)

    def __lt__(self, other):
        return self._cmp("<", other)

    def __eq__(self, other):  # builds a constraint, as spice_segmenter does
        return self._cmp("==", other)

    __hash__ = object.__hash__


@pdataclass(config=CFG, eq=False, repr=False)
class TargetedProperty(Property):
    observer: SpiceRefT
    target: SpiceBodyT
    light_time_correction: str = "NONE"


@pdataclass(config=CFG, eq=False, repr=False)
class Distance(TargetedProperty):
    _name: ClassVar[str] = "distance"
    unit: UnitT = Field(default="km", kw_only=True)

    def __call__(self, t: str) -> float:
        return float(np.linalg.norm(sp.spkpos(self.target.name, sp.str2et(t), "J2000", self.light_time_correction,
                                              self.observer.name)[0]))


@pdataclass(config=CFG, eq=False, repr=False)
class PhaseAngle(TargetedProperty):
    _name: ClassVar[str] = "phase_angle"
    third_body: SpiceBodyT = "SUN"
    unit: UnitT = Field(default="deg", kw_only=True)


@pdataclass(config=CFG, eq=False, repr=False)
class PushbroomRepetitionTime(TargetedProperty):
    _name: ClassVar[str] = "pushbroom_repetition_time"
    unit: UnitT = Field(default="s", kw_only=True)
    overlap: float = Field(default=0.10, ge=0.0, lt=1.0, kw_only=True)
    fov_axis: str = Field(default="long", pattern="^(long|short)$", kw_only=True)


@pdataclass(config=CFG, eq=False, repr=False)
class ScalarConstant(Property):
    _name: ClassVar[str] = "scalar_constant"
    value: QuantityT = None


# --- 3. recursive tagged union built from the registry -------------------------------------------------------------
def _tag(v):
    return v.get("type") if isinstance(v, dict) else type(v).__name__


def node_union():
    members = [Annotated[cls, Tag(name)] for name, cls in REGISTRY.items()] + [Annotated["Constraint", Tag("Constraint")]]
    return Annotated[Union[tuple(members)], Discriminator(_tag)]


@pdataclass(config=CFG, eq=False, repr=False)
class Constraint:
    left: Any
    operator: str
    right: Any

    def __and__(self, other):
        return Constraint(left=self, operator="&", right=other)


def to_wire(obj) -> Any:
    """Tagged dict with the class name as "type" (the cattrs format)."""
    if isinstance(obj, (Property, Constraint)):
        ta = TypeAdapter(type(obj))
        d = ta.dump_python(obj, mode="json", exclude={"left", "right"} if isinstance(obj, Constraint) else None)
        if isinstance(obj, Constraint):
            d = {"left": to_wire(obj.left), "operator": obj.operator, "right": to_wire(obj.right)}
        return {"type": type(obj).__name__, **d}
    return obj


def from_wire(d: dict):
    d = dict(d)
    kind = d.pop("type")
    if kind == "Constraint":
        return Constraint(left=from_wire(d["left"]), operator=d["operator"], right=from_wire(d["right"]))
    return REGISTRY[kind](**d)


def json_schema() -> dict:
    """Schema of the whole tree: each node a tagged object."""
    defs = {}
    for name, cls in REGISTRY.items():
        s = TypeAdapter(cls).json_schema(mode="serialization")
        s.setdefault("properties", {})["type"] = {"const": name}
        s["required"] = ["type", *s.get("required", [])]
        defs[name] = s
    node = {"oneOf": [{"$ref": f"#/$defs/{n}"} for n in [*REGISTRY, "Constraint"]], "discriminator": {"propertyName": "type"}}
    defs["Constraint"] = {"type": "object", "required": ["type", "left", "operator", "right"],
                          "properties": {"type": {"const": "Constraint"}, "left": {"$ref": "#/$defs/Node"},
                                         "operator": {"enum": [">", "<", "==", "&", "|"]}, "right": {"$ref": "#/$defs/Node"}}}
    defs["Node"] = node
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$ref": "#/$defs/Node", "$defs": defs}


if __name__ == "__main__":
    # 1. positional args and operators
    d = Distance("JUICE_JANUS", "CALLISTO")
    c = (PhaseAngle("JUICE_JANUS", "CALLISTO") < "90 deg") & (d > "5e4 km") & (
        PushbroomRepetitionTime("JUICE_JANUS", "CALLISTO", overlap=0.2) < "5.7 min")
    print("1. constraint built:", type(c).__name__, "| == builds:", type(d == "1 km").__name__)
    print("   value:", d("2032-12-28T07:27"), "km; validation:", end=" ")
    try:
        PushbroomRepetitionTime("JUICE_JANUS", "CALLISTO", overlap=1.5)
    except Exception as e:
        print(type(e).__name__, str(e).splitlines()[1].strip())
    # 2./3. wire format, round trip, schema
    w = to_wire(c)
    print("3. wire:", json.dumps(w)[:220], "...")
    back = from_wire(json.loads(json.dumps(w)))
    print("   round trip equal wire:", to_wire(back) == w)
    sch = json_schema()
    print("   schema defs:", list(sch["$defs"]), "| PushbroomRepetitionTime props:",
          json.dumps(sch["$defs"]["PushbroomRepetitionTime"]["properties"])[:300])
    try:
        import jsonschema

        jsonschema.validate(w, sch)
        bad = json.loads(json.dumps(w)); bad["left"]["left"]["right"]["value"] = 3
        jsonschema.validate(bad, sch)
    except ImportError:
        print("   (jsonschema not installed)")
    except Exception as e:
        print("   schema rejects a bad document:", type(e).__name__)
    # 4. cost
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        Distance("JUICE_JANUS", "CALLISTO")
    t_p = (time.perf_counter() - t0) / n * 1e6
    from spice_segmenter import Distance as ADistance

    t0 = time.perf_counter()
    for _ in range(n):
        ADistance("JUICE_JANUS", "CALLISTO")
    t_a = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        d.observer
    t_get = (time.perf_counter() - t0) / n * 1e9
    print(f"4. construct: pydantic {t_p:.1f} us vs attrs {t_a:.1f} us; attribute read {t_get:.0f} ns")
