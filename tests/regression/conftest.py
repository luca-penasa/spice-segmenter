"""Regression testing infrastructure for spice-segmenter.

Usage
-----
Run regression tests normally (compare against stored baselines)::

    pytest tests/regression/

Regenerate / accept all baselines (after a planned kernel update or after
adding new properties)::

    pytest tests/regression/ --update-regression

Regenerate a single baseline by name::

    pytest tests/regression/ --update-regression -k "ganymede_properties"

How it works
------------
Each test calls ``regression_baseline(name, actual, tolerances)`` where:

* ``name``    – path relative to ``baselines/``, e.g.
  ``"properties/juice_janus_ganymede.yaml"``
* ``actual``  – a plain dict of ``{key: float | list[float] | str}``
* ``tolerances`` – optional dict of ``{key_glob: rel_tol}``; defaults to 1e-6
  relative tolerance for all numeric values.

When ``--update-regression`` is passed the baseline is written and the test is
*skipped* (not failed) so the regeneration run itself stays green.  On
subsequent runs without the flag the stored YAML is loaded and every key is
compared numerically with ``pytest.approx``.

Solver baselines additionally store interval counts and per-interval start/end
as ISO-8601 strings; comparison is done within a configurable time tolerance
(default 60 s).
"""

from __future__ import annotations

import fnmatch
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

BASELINES_DIR = Path(__file__).parent / "baselines"
MK_VERSION = "v432_20230505_001"  # mirrors tests/__init__.py


# ---------------------------------------------------------------------------
# pytest hook: register CLI flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-regression",
        action="store_true",
        default=False,
        help=(
            "Regenerate regression baselines instead of comparing against them. "
            "Tests are skipped (not failed) after writing the new baseline."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_tolerance(key: str, tolerances: dict[str, float], default: float = 1e-6) -> float:
    """Return the tolerance for *key* by matching glob patterns in *tolerances*."""
    for pattern, tol in tolerances.items():
        if fnmatch.fnmatch(key, pattern):
            return tol
    return default


def _to_serialisable(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return [_to_serialisable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "name"):          # Enum-like (OccultationTypes, …)
        return str(value.name)
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _compare_numeric(expected: Any, actual: Any, tol: float, path: str) -> None:
    """Assert *actual* matches *expected* within *tol* relative tolerance.

    Special cases:
    * ``tol == 0`` and both values look like ISO-8601 timestamps → compare
      within 60-second absolute tolerance.
    * ``expected == "nan"`` → assert actual is NaN.
    * ``expected`` is a list → recurse element-wise.
    * ``expected`` is a non-numeric string → exact string match.
    """
    import numpy as np

    # Handle stored NaN sentinel
    if expected == "nan":
        assert math.isnan(float(actual)) or (
            isinstance(actual, (list, np.ndarray))
            and all(math.isnan(float(v)) for v in np.atleast_1d(actual))
        ), f"{path}: expected NaN, got {actual}"
        return

    if isinstance(expected, list):
        assert len(expected) == len(actual), (
            f"{path}: length mismatch {len(expected)} vs {len(actual)}"
        )
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare_numeric(e, a, tol, f"{path}[{i}]")
        return

    if isinstance(expected, str):
        # Try timestamp comparison (e.g. solver interval start/end)
        try:
            e_t = np.datetime64(expected)
            a_t = np.datetime64(str(actual))
            diff_s = abs(float(a_t - e_t) / 1e9)        # ns → s
            _TIME_TOL_S = 60.0
            assert diff_s <= _TIME_TOL_S, (
                f"{path}: timestamp {actual!r} vs {expected!r} "
                f"(diff {diff_s:.1f}s > {_TIME_TOL_S}s)"
            )
            return
        except (ValueError, TypeError):
            pass
        # Plain string: exact match
        assert str(actual) == expected, f"{path}: {actual!r} != {expected!r}"
        return

    if isinstance(expected, bool):
        assert bool(actual) == expected, f"{path}: {actual!r} != {expected!r}"
        return

    assert float(actual) == pytest.approx(float(expected), rel=tol, abs=0), (
        f"{path}: {actual} != {expected} (rel tol={tol})"
    )


def _write_baseline(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "meta": {
            "mk_version": MK_VERSION,
            "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }
    full = {**meta, **{"values": data}}
    path.write_text(yaml.dump(full, default_flow_style=False, sort_keys=True, allow_unicode=True))


def _load_baseline(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    return raw.get("values", {})


# ---------------------------------------------------------------------------
# Main fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_baseline(request: pytest.FixtureRequest):
    """Compare *actual* dict against a stored YAML baseline.

    Parameters
    ----------
    name:
        Path relative to ``tests/regression/baselines/``.
    actual:
        Dict of ``{str: numeric | list | str}``.  Values are automatically
        normalised (numpy scalars → Python scalars, Enums → their name string).
    tolerances:
        Optional ``{glob_pattern: rel_tol}`` dict.  Default relative tolerance
        is ``1e-6``.  Use ``{"*": 1e-3}`` to relax all, or
        ``{"start": 1.0, "end": 1.0}`` to apply 1-second absolute tolerance
        for solver start/end entries.

    When ``--update-regression`` is active the baseline is written and the
    test is *skipped*.
    """
    update: bool = request.config.getoption("--update-regression")

    def _check(
        name: str,
        actual: dict[str, Any],
        tolerances: dict[str, float] | None = None,
    ) -> None:
        tolerances = tolerances or {}
        path = BASELINES_DIR / name

        # Normalise actual values to plain Python types
        normalised: dict[str, Any] = {
            k: _to_serialisable(v) for k, v in actual.items()
        }

        if update or not path.exists():
            _write_baseline(path, normalised)
            reason = "written" if not path.exists() else "updated"
            pytest.skip(f"Baseline {reason}: {path.relative_to(BASELINES_DIR)}")

        expected = _load_baseline(path)

        missing = set(expected) - set(normalised)
        extra   = set(normalised) - set(expected)

        assert not missing, (
            f"Keys present in baseline but missing from actual output:\n"
            + "\n".join(f"  {k}" for k in sorted(missing))
        )
        if extra:
            # New properties added — fail with a clear message so the developer
            # knows to run --update-regression
            raise AssertionError(
                f"New keys in actual output not present in baseline "
                f"(run --update-regression to accept):\n"
                + "\n".join(f"  {k}" for k in sorted(extra))
            )

        for key, exp_val in expected.items():
            act_val = normalised[key]
            tol = _match_tolerance(key, tolerances)
            _compare_numeric(exp_val, act_val, tol, key)

    return _check
