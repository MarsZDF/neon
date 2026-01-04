# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of elemental-neon
- `neon.compare` module for approximate equality comparisons
  - `near()`, `near_zero()`, `is_integer()` functions
  - `less_or_near()`, `greater_or_near()`, `compare()` functions
  - `all_near()`, `near_many()` batch operations
- `neon.clamp` module for value snapping and clamping
  - `to_zero()`, `to_int()`, `to_value()` functions
  - `to_range()`, `to_values()` functions
  - `to_zero_many()` batch operation
- `neon.safe` module for safe arithmetic
  - `div()`, `div_or_zero()`, `div_or_inf()` safe division
  - `mod()`, `sqrt()`, `log()`, `pow()` safe operations
  - `sum_exact()`, `mean_exact()` Kahan summation
- `neon.ulp` module for ULP-based operations
  - `of()`, `diff()`, `near()` ULP functions
  - `next()`, `prev()`, `add()` ULP manipulation
- Exception hierarchy: `NeonError`, `InvalidValueError`, `EmptyInputError`
- Comprehensive test suite with pytest
- Full type hints and docstrings
- Zero-dependency implementation

## [0.1.0] - 2026-01-04

### Added
- Initial beta release
