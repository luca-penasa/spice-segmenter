# Migrating the core to pydantic: assessment

*Status: investigation, October 2026. Nothing migrated yet.*

## Recommendation

Migrate the core classes from attrs to **pydantic dataclasses** (`pydantic.dataclasses.dataclass`), not to
`pydantic.BaseModel`, and do it incrementally behind a compatibility layer. The public API stays as it is
(positional construction, operator overloading, the DSL, the YAML format); what is gained:

- **a JSON Schema for free** for every property, constant and the recursive constraint tree, built from the registry,
  with field constraints (`ge`, `lt`, `pattern`) and descriptions included;
- **validation** at construction from declared types instead of ad-hoc converters;
- **serialization without cattrs**: `TypeAdapter.dump_python` / `validate_python`, keeping today's wire format
  (`{"type": "<ClassName>", ...}`), so existing JSON/YAML stays readable;
- a clean basis for **database records** (below).

`BaseModel` is the wrong base for properties: it takes keyword arguments only (every `Distance("JUICE_JANUS",
"CALLISTO")` in jana and the notebooks would break) and defines `__eq__`, which the segmenter overloads to build
constraints. Pydantic dataclasses keep positional arguments and accept `eq=False`.

## What the core is today

- 11,700 lines; **107 attrs classes**, 101 of them `@define(repr=False, order=False, eq=False)` (the operators
  `<`, `>`, `==`, `&` build constraints, so attrs equality is off).
- **50 field converters**, almost all of six kinds: pint units (16), SPICE references (`SpiceRef`, `SpiceBody`,
  `SpiceInstrument`, `SpiceFrame`, `as_spice_ref`: 18), timestamps / timedeltas (3), a few lambdas (8).
- attrs introspection in 7 modules: `attrs.fields` (8), `attrs.NOTHING` (10), `attrs.evolve` (2), `attrs.Factory`
  (2): `core/registry.py`, `io/dsl.py`, `io/yaml_io.py`, `support/context.py`, `support/config.py`,
  `properties/component_selector.py`, `core/property.py`.
- serialization: `support/serialization.py`, cattrs with a tagged union on the class name (181 lines).
- downstream: jana's obsdefs compiler fingerprints constraints with `attrs.fields(type(obj))`
  (`jana/planning/obsdefs/compiler.py`); `TimeSegment` derives from ptr-editor's `time_segments.TimeSegmentMixin`,
  which itself calls `attrs.fields(type(self))`.

## Prototype

`docs/design/pydantic_prototype.py` (run: `uv run python docs/design/pydantic_prototype.py`) rebuilds `Property`, `TargetedProperty`, `Distance`, `PhaseAngle`,
`PushbroomRepetitionTime`, `ScalarConstant` and `Constraint` as pydantic dataclasses, against the real kernels:

| check | result |
|---|---|
| positional construction, `<` `>` `==` `&` building constraints | works (`eq=False`, `__hash__ = object.__hash__`) |
| abstract base (`ABC`), mixins, `dataclasses.fields`, `replace` (re-validates), `deepcopy`, `pickle` | all work |
| SPICE / pint fields | `Annotated[..., BeforeValidator, PlainSerializer, WithJsonSchema]`, one alias per kind |
| field constraints (`overlap` in [0, 1), `fov_axis` pattern) | enforced, and present in the schema |
| wire format | same tagged tree as cattrs; JSON round trip identical |
| JSON Schema of the tree | generated from the registry (one `$def` per class, `oneOf` with a `type` discriminator); `jsonschema` accepts a real constraint and rejects a malformed one |
| cost | construction 1.1-2x attrs over two runs (50 vs 44 µs, 159 vs 76 µs; both dominated by SPICE name lookups); attribute read 30-90 ns. Evaluation is SPICE-bound and unchanged |

Points that need care:

- **Defaults are not validated by default** in pydantic: set `validate_default=True` (attrs converters run on
  defaults; without it a unit default stays a string).
- **Tagged union over a dynamic registry**: build it at runtime from `property_registry`
  (`Annotated[Union[...], Discriminator(callable)]` with `Tag(class name)`); regenerate after plug-in modules
  register properties.
- **attrs-only features**: `attrs.NOTHING` (use `dataclasses.MISSING` / pydantic `PydanticUndefined`),
  `attrs.Factory` (`Field(default_factory=...)`), `attrs.evolve` (`dataclasses.replace`).
- **`TimeSegment`**: keep it on attrs until `time_segments.TimeSegmentMixin` (ptr-editor) stops calling
  `attrs.fields`; or make the mixin field-agnostic first (a small helper that reads `attrs.fields` or
  `dataclasses.fields`). jana's fingerprint needs the same helper.
- slots: attrs classes are slotted today; pydantic dataclasses can be (`slots=True`), but the one class using
  `cached_property` must stay unslotted.

## Plan

1. **Foundations** (no behaviour change): `support/types.py` with the annotated SPICE / pint / time types; a
   `segmenter_dataclass` decorator (`config`, `eq=False`, `repr=False`); a field-introspection helper used everywhere
   instead of `attrs.fields` (works for both while classes are mixed).
2. **Core**: `Property`, `TargetedProperty`, `ConstraintBase` / `Constraint`, `ops/` (constants, `Inverted`,
   `WrappedConstraint`), `ComponentSelector`; new `support/serialization.py` on `TypeAdapter` producing the current
   wire format, with `create_property_converter` / `structure_constraint` / `unstructure_constraint` kept as thin
   wrappers; `json_schema()` and a CLI entry (`spice-segmenter schema`).
3. **Property modules** (geometry 38, observation 17, coordinates 12, others): mechanical, by a codemod
   (decorator, `field(...)` -> `Field(...)`, converter -> annotated type); the test suite (310 tests, including
   serialization and the property regression baselines) after each module.
4. **I/O and registry**: `io/dsl.py`, `io/yaml_io.py`, `core/registry.py`, `support/context.py`, `support/config.py`
   on the helper; drop cattrs.
5. **Downstream**: jana fingerprint on the helper, jana parity test and test suite, ptr-editor's mixin, the
   notebooks.

Size: about 3 working days, most of it steps 2-3; each step lands separately with the suite green.

## Databases

The schema benefit is concrete for two kinds of records, which can be pydantic `BaseModel` / SQLModel tables
whatever the core uses:

- **definitions**: an opportunity (name, target, group, tags, description) with its constraint stored as a JSON
  document (PostgreSQL `jsonb`, SQLite `JSON`) and validated against the generated schema; the DSL string can be
  stored alongside for people;
- **results**: solved segments as rows (opportunity, start, end, kernel set / metakernel, solver version), the part
  queried most (by time, flyby, target).

Constraint trees are recursive and open-ended (new properties register themselves): a relational table per property
would be brittle. The tree belongs in one JSON column, validated by the schema; the results belong in relational
tables.
