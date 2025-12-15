# Copilot Instructions for spice-segmenter

## Project Overview
**spice-segmenter** is a Python library for trajectory event segmentation using NASA's SPICE toolkit. It bridges high-level mission operations (distance constraints, occultations, target visibility) with low-level SPICE ephemeris calculations.

## Architecture

### Core Pattern: Property-based Constraints
Everything revolves around three core abstractions:

1. **Properties** (`property_base.Property`): Compute single scalar values at a given time
   - Examples: `Distance`, `PhaseAngle`, `AngularSize`, `Occultation`
   - Implement `__call__(time)` to evaluate at specific times
   - Must define `unit` property (using `pint` for unit tracking)

2. **Constraints** (`constraint.Constraint`): Boolean comparisons of properties
   - Built via operators: `Distance(...) < '100 km'`, `Occultation(...) == OccultationTypes.FULL`
   - Support composition: `constraint1 & constraint2`, `~constraint1`
   - Solve via `.solve(scenario)` to find time windows where constraint is True

3. **Collections** (`collections.py`): Convenience groupings
   - `TargetProperties`: Groups `distance`, `phase_angle`, `angular_size`, etc. for a target
   - `OccultationProperties`: Groups occultations by multiple occulting bodies

### Data Model
- **SpiceWindow**: Time intervals representing solution windows (backed by SPICE gfsubc)
- **Properties** call SPICE functions (vectorized via `@vectorize` decorator)
- **Units**: All properties use `pint.Unit`; constraints auto-convert via `UnitAdaptor`

## Critical Implementation Details

### Circular Import Prevention
**NEVER import from the top-level `spice_segmenter` package inside core modules.** Use relative imports instead:
- ✅ **DO**: `from .trajectory_properties import Distance`
- ❌ **DON'T**: `from spice_segmenter import Distance` (causes circular imports during `__init__.py` loading)
- **Reason**: `__init__.py` imports from `collections.py`, which was importing from package level

### Decorators & Registration
- `@declare(name="...", unit=...)`: Registers property in `PROPERTIES_REGISTRY`, auto-generates `__repr__` and `__str__`
- `@vectorize`: Wraps functions to handle both scalar and array times (returns numpy arrays)
- These decorators are applied at class definition time

### Constraint Optimization (New Feature)
The `constraint_optimizer.py` module automatically replaces slow properties with faster equivalents:
- **Problem**: `TargetSizeOnSensor` and `AngularSize` call expensive SPICE event-finding (gfuds)
- **Solution**: Transform to `Distance` comparisons (single geometric calculation)
- **API**: `constraint.solve(scenario, optimize=True)` or `optimize_constraint(constraint)`
- **Transformers**: `PropertyTransformer` base class allows adding new optimization strategies

## Development Workflow

### Setup
```bash
uv sync --all-extras  # Install all dependencies in virtual environment
source .venv/bin/activate  # Activate venv (.venv\Scripts\activate on Windows)
```

### Testing
```bash
uv run pytest                  # Run all tests
uv run pytest tests/test_properties.py::test_distance -v
```

### Common Commands
```bash
just bump patch         # Bump version, update CHANGELOG, create tag (use major/minor/patch)
uv run mypy src         # Type checking
uv run ruff check src tests
```

### Pre-commit
Pre-commit hooks run automatically on commit (`pre-commit install` enables them). They run:
- `ruff format` (auto-formatting)
- `ruff check` (linting)
- `mypy` (type checking)

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `src/spice_segmenter/__init__.py` | Package entry point; manages import order to avoid circular deps |
| `property_base.py` | Abstract `Property` base class; defines interface for all observables |
| `trajectory_properties.py` | Concrete properties: `Distance`, `PhaseAngle`, `AngularSize`, `Occultation`, etc. |
| `constraint.py` | `Constraint` class; builds/composes boolean conditions; implements `.solve()` |
| `collections.py` | `TargetProperties`, `OccultationProperties` for convenience APIs |
| `constraint_optimizer.py` | Transforms slow properties to fast equivalents (e.g., `TargetSizeOnSensor` → `Distance`) |
| `spice_window.py` | Represents solution time windows; wraps SPICE ET windows |
| `coordinates.py` | Coordinate systems: Spherical, Cylindrical, Geodetic, RaDec, etc. |
| `serialization.py` | `cattrs` converters for JSON serialization of constraints/properties |

## Common Patterns

### Adding a New Property
1. Define class inheriting from `Property` or `TargetedProperty`
2. Implement `__call__(time)` → evaluates SPICE function(s) at time(s)
3. Add `@declare(name="...", unit=...)` decorator
4. Use `@vectorize` if calling SPICE functions that need array handling
5. Export in `__init__.py`

### Adding a New Constraint Type
1. Subclass `ConstraintBase` (not often needed; most use comparison operators on properties)
2. Implement abstract `left`, `right` properties and `type` property
3. The `Constraint` class handles most use cases via operator overloading

### Adding an Optimization Strategy
1. Subclass `PropertyTransformer` in `constraint_optimizer.py`
2. Implement `can_transform(property)` and `transform(property)` methods
3. Register in `ConstraintOptimizer.transformers` list

## Testing Conventions
- Test modules use `tour_config` fixture (test/`__init__.py`) providing SPICE kernels and ephemeris
- Tests compare against precomputed reference data: `tc[time].dist[0]`, `tc[time].phase[0]`
- Use `pytest.approx()` for floating-point comparisons
- Import directly from submodules, not via package level (avoids circular imports in tests)

## Dependencies & External APIs
- **spiceypy**: SPICE kernel calls; vectorized via callbacks (SpiceUDFUNS, SpiceUDFUNB)
- **attrs**: Class definitions with `@define`; use `field()` for custom converters
- **pint**: Unit system; all properties expose units
- **planetary-coverage**: High-level SPICE wrappers (SpiceBody, SpiceSpacecraft)
- **cattrs**: Structure/unstructure for serialization
- **loguru**: Logging; disabled by default; enable with `log_enable()`

## Common Pitfalls
1. **Circular imports**: Always use relative imports in core modules
2. **Unit mismatches**: Ensure constraint comparisons have compatible units (handled by UnitAdaptor)
3. **Time types**: Accept `TIMES_TYPES` (scalar, array, or datetime); use `@vectorize` to handle all
4. **SPICE kernels**: Tests require kernels via `tour_config`; see `tests/__init__.py`
