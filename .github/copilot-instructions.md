# Copilot Instructions for spice-segmenter

## Project Overview
**spice-segmenter** is a Python library for trajectory event segmentation using NASA's SPICE toolkit. It bridges high-level mission operations (distance constraints, occultations, target visibility) with low-level SPICE ephemeris calculations.

## Architecture

### Core Pattern: Property-based Constraints
Everything revolves around three core abstractions:

1. **Properties** (`core.property.Property`): Compute single scalar values at a given time
   - Examples: `Distance`, `PhaseAngle`, `AngularSize`, `Occultation`
   - Implement `__call__(time)` to evaluate at specific times
   - Must define `unit` property (using `pint` for unit tracking)
   - Subclasses: `BooleanProperty` (returns True/False), `TargetedProperty` (operates on celestial bodies)

2. **Constraints** (`core.constraints.Constraint`): Boolean comparisons of properties
   - Built via operators: `Distance(...) < '100 km'`, `Occultation(...) == OccultationTypes.FULL`
   - Support composition: `constraint1 & constraint2`, `~constraint1`
   - Solve via `.solve(scenario)` to find time windows where constraint is True
   - Solved using `constraint_solver.constraint_solver.ConstraintSolver` (SPICE event finding)

3. **Collections** (`collections/property_collections.py`): Convenience groupings
   - `TargetProperties`: Groups `distance`, `phase_angle`, `angular_size`, etc. for a target
   - `OccultationProperties`: Groups occultations by multiple occulting bodies

### Data Model
- **SpiceWindow** (`core.spice_window`): Time intervals as SPICE ET pairs (backed by SPICE gfsubc)
- **Properties** call SPICE functions (vectorized via `@vectorize` decorator)
- **Units**: All properties use `pint.Unit`; constraints auto-convert via `ops.unit_adapter.UnitAdaptor`

## Critical Implementation Details

### Circular Import Prevention
**NEVER import from the top-level `spice_segmenter` package inside core/ops/properties modules.** Use relative imports instead:
- ✅ **DO**: `from ..core.property import Property` or `from .observation_properties import Distance`
- ❌ **DON'T**: `from spice_segmenter import Distance` (causes circular imports during `__init__.py` loading)
- **Reason**: `__init__.py` imports from `collections/`, `constraint_solver/`, and other modules; importing back creates cycles

### Property Registration & Decorators
- `@declare(name="...", unit=...)`: Registers property metadata (name, unit, type) via `PropertyMeta` metaclass
- `@vectorize`: Wraps `__call__` to handle both scalar and array times (returns numpy arrays)
- `PropertyMeta` registry: Accessible via `PropertyMeta.registry[name]` for dynamic property lookups
- Applied at class definition time; no runtime overhead

### Module Organization
**Core (`core/`)**: Abstract interfaces and fundamental concepts
- `property.py`: `Property`, `BooleanProperty` abstract base classes
- `constraints.py`: `Constraint`, `ConstraintBase` for boolean logic
- `spice_window.py`: Time interval representation

**Properties (`properties/`)**: Concrete property implementations by category
- `observation_properties.py`: `Distance`, `PhaseAngle`, `AngularSize`, `TargetSizeOnSensor`
- `surface_properties.py`: `SubObserverPoint`, `SubObserverPointVelocity`, `SurfaceIlluminationAngles`
- `visibility_properties.py`: `BodyFOVVisibility`, `SubObserverIsInDaylight`
- `occultation_types.py`: `OccultationTypes` enum and `Occultation` property
- `coordinates.py`: Coordinate systems (`SphericalCoordinates`, `RaDecCoordinates`, etc.)

**Constraint Solving (`constraint_solver/`)**: Event finding logic
- `constraint_solver.py`: `ConstraintSolver` base class and implementations
  - Handles SPICE event-finding (gfsubc, gfuds, etc.)
  - Coordinates step sizes and refinement logic
  - Returns `SpiceWindow` with solution intervals

**Operations (`ops/`)**: Utilities and constraint building
- `unit_adapter.py`: `UnitAdaptor` for unit conversion in constraints
- `constraint_operations.py`: `Inverted`, `MinMaxConstraint`
- `constant_values.py`: `Constant`, `BoolConstant` for constraint right-hand sides

**Optimization (`optimizers/`)**: Performance improvements
- `constraint_optimizer.py`: `PropertyTransformer` base class and implementations
  - `TargetSizeOnSensorToDistance`: Converts slow gfuds calls to distance comparisons
  - `ConstraintOptimizer`: Orchestrates transformation pipeline

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
uv run mypy src         # Type checking (requires type stubs)
uv run ruff check src tests  # Linting
uv run ruff format src tests # Auto-format
```

### Pre-commit Hooks
Pre-commit hooks run automatically on commit (`pre-commit install` enables them):
- `ruff format` (auto-formatting)
- `ruff check` (linting with fixes)
- `mypy` (type checking)

Run manually: `pre-commit run --all-files`

### SPICE Kernels & Test Data
- Tests use `tour_config` fixture from `tests/__init__.py`
- Loads kernels via `SpiceManager` from `quick-spice-manager` package
- Provides `tour_config.coverage` (start/end times) and `tc[time]` reference data
- **Critical**: Tests will fail without kernels; `tour_config` handles kernel loading automatically

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `src/spice_segmenter/__init__.py` | Package entry point; manages import order to avoid circular deps (see comments on import sequence) |
| `core/property.py` | Abstract `Property` and `BooleanProperty` base classes; defines interface for all observables |
| `core/constraints.py` | `Constraint` class; builds/composes boolean conditions; implements `.solve()` method |
| `core/spice_window.py` | Time interval representation; wraps SPICE ET windows (start/end pairs) |
| `properties/observation_properties.py` | Core properties: `Distance`, `PhaseAngle`, `AngularSize`, `TargetSizeOnSensor` |
| `properties/surface_properties.py` | Surface geometry: `SubObserverPoint`, `SurfaceIlluminationAngles`, etc. |
| `properties/visibility_properties.py` | Visibility: `BodyFOVVisibility`, `SubObserverIsInDaylight` |
| `properties/occultation_types.py` | `OccultationTypes` enum and `Occultation` property implementation |
| `properties/coordinates.py` | Coordinate systems: `SphericalCoordinates`, `CylindricalCoordinates`, `RaDecCoordinates`, etc. |
| `constraint_solver/constraint_solver.py` | SPICE event-finding solver; implements GF (geometry finder) event detection |
| `ops/unit_adapter.py` | `UnitAdaptor` for automatic unit conversion in constraint comparisons |
| `ops/constraint_operations.py` | Constraint combinators: `Inverted` (~constraint), `MinMaxConstraint` (&& logic) |
| `optimizers/constraint_optimizer.py` | Performance optimization: transforms slow properties to fast equivalents |
| `collections/property_collections.py` | High-level convenience APIs: `TargetProperties`, `OccultationProperties` |
| `support/decorators.py` | `@declare`, `@vectorize`, and `PropertyMeta` metaclass for property registration |
| `support/serialization.py` | `cattrs` converters for JSON serialization of constraints/properties |
| `support/config.py` | Global configuration: `solver_step`, solver tolerances, etc. |

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
