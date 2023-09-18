# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Added SHOW_PROGRESSBAR variable in module init, to disable the progress bar during event finding. This will probably need to be also read from environment variables or config files.
- Added a call method to SpiceWindow class, not it can be called the same as other properties.
- Sub observer point as a Vector Property

### Fixed
- Bug in the calculation of some coordinates properties.
