# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
