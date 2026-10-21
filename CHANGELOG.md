# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The data classes (properties, constraints, constants, solvers, `SpiceContext`, `Config`, `TimeSegment`, `TimeSegmentsCollection`, ...) are pydantic dataclasses instead of attrs classes; attrs and cattrs are no longer dependencies, pydantic >= 2.11 is. Construction (positional arguments, operators building constraints) and the JSON / YAML formats are unchanged.
  - Field introspection goes through `spice_segmenter.support.fields` (`fields(cls)`, `evolve(obj, **changes)`) instead of `attrs.fields` / `attrs.evolve`; code reading `attrs.fields(type(prop))` (e.g. fingerprints in downstream packages) needs the same change.
  - Invalid or unknown constructor arguments raise `pydantic.ValidationError` (a `ValueError`) instead of `TypeError`.
  - `TimeSegment` mutation raises `dataclasses.FrozenInstanceError` (still an `AttributeError`) instead of `attrs.exceptions.FrozenInstanceError`.
  - `create_property_converter()` returns a small converter with the same `unstructure` / `structure` methods instead of a `cattrs.Converter`.
  - Boolean constants serialize as JSON booleans (`"_value": true`); the former `"True"` / `"False"` strings are still read.

### Added

- `spice_segmenter.support.serialization.json_schema()`: JSON Schema of the serialized property / constraint tree.

### Fixed

- JSON round trip of `Inverted`, `MinMaxConstraint` and `ComponentSelector` (unit overrides included); they could be written but not read back.
- `property_registry["component_selector"]` returned a stale, field-less class object left over by the attrs slots rewrite.
- A `BoolConstant` serialized as `"False"` was read back as `True`.

- Coordinate transforms (`as_latitudinal`, `as_spherical`, `as_cylindrical`, `as_radec`, `as_geodetic`, `as_planetographic`) returned every angle converted to radians twice (57x too small): they were registered with degree compute units while the SPICE `rec*` routines return radians. Constraints on them (e.g. a latitude or ring opening angle against a degree threshold) were wrong accordingly.
- `as_geodetic` failed: `recgeo` was called without the body's equatorial radius and flattening.
- Constraint language: context keys a property does not take (e.g. `target` for `occultation` or the ring-ansa properties) are no longer passed to its constructor, so one context can be shared across all leaves.
- Vectorised `Occultation` evaluation passes frame names as strings to `cyice.occult_v` (it raised `TypeError: expected bytes, SpiceFrame found`).
- Test setup uses the quick-spice-manager 0.2 API (`QuickSpiceManager(...).get_tour_config(instrument=None)`); requires quick-spice-manager >= 0.2.0.

## 0.0.6 - 2026-06-06

### Changed

- Full experimental refactor of the codebase, strongly influenced by copilot work. Still experimental.

### Added

- new target size on sensor property

## 0.0.5 - 2024-11-08

### Added

- spice-based solver for boolean properties + boolean props interface
- boresightintesect property
- spice-based solver for generic scalar properties, first implementation.

### Changed

- now relying on quick-spice-manager to download spice kernels
- Constants can be instantiated from pint.Quantity so thata we can write `phase < Quantity(22, 'deg')`

## [0.0.4] - 2024-01-16

### Changed

- the configuration mechanisms now is a single dict

### Added

- Angular size property for a target
- Top level imports now available for some classes
- Angular separation property added: can also be used in searches
- Fov visibility search implemented. still to be improved.

### Removed

- Some boilerplate code by introducing a decorator to declare properties.

## [0.0.3] - 2023-09-18

### Added

- Added SHOW_PROGRESSBAR variable in module init, to disable the progress bar during event finding. This will probably need to be also read from environment variables or config files.
- Added a call method to SpiceWindow class, not it can be called the same as other properties.
- Sub observer point as a Vector Property

### Fixed

- Bug in the calculation of some coordinates properties.

[Unreleased]: https://github.com/JANUS-JUICE/spice_segmenter/compare/0.0.4...master
[0.0.4]: https://github.com/JANUS-JUICE/spice_segmenter/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/luca-penasa/spice_segmenter/tree/0.0.3
