"""Evaluator: dispatches property evaluation through a ComputationEngine.

The global ``get_evaluator()`` function returns a lazily-initialised singleton
that owns a :class:`SpiceEngine` pre-loaded with all registered compute
functions.  Call this instead of instantiating ``Evaluator`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pint
import spiceypy.utils.callbacks

if TYPE_CHECKING:
    from ..core.property import Property
    from .spice_engine import SpiceEngine

from ..support.time_types import TIMES_TYPES


def _apply_unit_conversion(value: Any, compute_unit: Any, desired_unit: Any) -> Any:
    """Convert *value* from *compute_unit* to *desired_unit*.

    Rules:
    * If *compute_unit* is ``None`` or equal to *desired_unit* → pass through.
    * Tuple units (vector properties) → per-component conversion, stacked on
      the last axis so the result shape matches the input.
    * Scalar units → pint magnitude conversion.
    """
    if compute_unit is None:
        return value
    if isinstance(compute_unit, tuple):
        if compute_unit == desired_unit:
            return value
        # Value is (..., N) shaped; split along last axis and convert each component.
        n = len(compute_unit)
        if not isinstance(desired_unit, tuple) or len(desired_unit) != n:
            return value  # cannot convert dissimilar tuple shapes
        parts = []
        for i, (cu, du) in enumerate(zip(compute_unit, desired_unit)):
            if cu == du:
                parts.append(value[..., i])
            else:
                parts.append(
                    pint.Quantity(value[..., i], cu).to(du).magnitude
                )
        return np.stack(parts, axis=-1)
    # Scalar unit
    if compute_unit == desired_unit:
        return value
    try:
        return pint.Quantity(value, compute_unit).to(desired_unit).magnitude
    except Exception:
        return value  # incompatible units — return raw (callers should avoid this)


class Evaluator:
    """Dispatches property evaluation through a :class:`SpiceEngine`.

    This is the single choke-point through which all property values flow.  It
    is the right place to add future cross-cutting concerns such as chunked
    parallel evaluation or caching.

    Fallback chain for scalar evaluation (highest priority first):

    1. Engine lookup by ``type(prop)`` (MRO walk).
    2. ``prop._call_scalar(time_et)`` — transitional path for properties that
       have not yet been migrated to engine registration.
    3. ``prop(time_et)`` — last resort for ``@vectorize``-decorated legacy
       properties that override ``__call__`` directly.
    4. :exc:`NotImplementedError` with a descriptive message.
    """

    def __init__(self, engine: SpiceEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate(self, prop: Property, time: TIMES_TYPES) -> Any:
        """Evaluate *prop* at one or more times, converting to ``prop.unit``."""
        from ..core.property import _bulk_et
        from ..support.spice_utilities import et as _et
        from ..support.config import get_active_config

        is_array = hasattr(time, "__len__") and not isinstance(time, str)
        if not is_array:
            return self.evaluate_scalar(prop, float(_et(time)))

        times_et = _bulk_et(time)
        if get_active_config().use_vectorized_calls:
            return self.evaluate_vector(prop, times_et)

        sig = getattr(prop, "_vector_output_shape", None)
        return np.vectorize(
            lambda t: self.evaluate_scalar(prop, t), signature=sig
        )(times_et)

    def evaluate_scalar_raw(self, prop: Property, time_et: float) -> Any:
        """Evaluate *prop* at a single ET, returning the value in *compute_unit*.

        No unit conversion is applied.  This is the path used by SPICE GF
        callbacks and by :meth:`compute_as_spice_function` so that the raw
        physical value (in the unit the compute function naturally returns) is
        always consistent with the ``refval`` the solver passes to SPICE.

        Fallback chain (same as :meth:`evaluate_scalar` but without conversion):
        1. Engine scalar fn.
        2. ``prop._call_scalar(time_et)`` (transitional).
        3. ``prop(time_et)`` (``@vectorize`` legacy — returns user-unit value).
        4. :exc:`NotImplementedError`.
        """
        try:
            return self._engine.evaluate_scalar(prop, time_et)
        except KeyError:
            pass

        try:
            return prop._call_scalar(time_et)  # type: ignore[attr-defined]
        except (NotImplementedError, AttributeError):
            pass

        # @vectorize legacy path: the property overrides __call__ directly.
        # Bypass Property.__call__ (which would re-enter the evaluator) by
        # invoking the method on the concrete type, which is the vectorised fn.
        try:
            return type(prop).__call__(prop, time_et)
        except Exception:
            pass

        raise NotImplementedError(
            f"{type(prop).__name__} has no registered compute function and no "
            f"_call_scalar implementation.  Register one with:\n"
            f"  engine.register({type(prop).__name__}, scalar_fn=your_fn, compute_unit=...)"
        )

    def evaluate_scalar(self, prop: Property, time_et: float) -> Any:
        """Evaluate *prop* at a single ET and convert to ``prop.unit``."""
        raw = self.evaluate_scalar_raw(prop, time_et)
        compute_unit = self._engine.get_compute_unit(type(prop))
        desired = getattr(prop, "unit", None)
        return _apply_unit_conversion(raw, compute_unit, desired)

    def evaluate_vector_raw(self, prop: Property, times_et: np.ndarray) -> np.ndarray:
        """Evaluate *prop* at an array of ETs, returning values in *compute_unit*."""
        try:
            return self._engine.evaluate_vector(prop, times_et)
        except KeyError:
            pass

        try:
            return prop._call_vector(times_et)  # type: ignore[attr-defined]
        except (NotImplementedError, AttributeError):
            pass

        raise NotImplementedError(
            f"{type(prop).__name__} has no registered compute function and no "
            f"_call_vector implementation.  Register one with:\n"
            f"  engine.register({type(prop).__name__}, scalar_fn=your_fn, compute_unit=...)"
        )

    def evaluate_vector(self, prop: Property, times_et: np.ndarray) -> np.ndarray:
        """Evaluate *prop* at an array of ETs and convert to ``prop.unit``."""
        raw = self.evaluate_vector_raw(prop, times_et)
        compute_unit = self._engine.get_compute_unit(type(prop))
        desired = getattr(prop, "unit", None)
        return _apply_unit_conversion(raw, compute_unit, desired)

    # ------------------------------------------------------------------
    # SPICE GF callback wrappers (used by constraint_solver)
    # ------------------------------------------------------------------

    def as_spice_function(
        self, prop: Property
    ) -> spiceypy.utils.callbacks.SpiceUDFUNS:
        """Wrap *prop* as a SPICE ``UDFUNS`` scalar callback (raw/native units).

        The returned object can be passed directly to ``spiceypy.gfuds``.
        Values are in *compute_unit* — no user-unit conversion is applied.
        """
        return spiceypy.utils.callbacks.SpiceUDFUNS(
            lambda t: float(self.evaluate_scalar_raw(prop, float(t)))
        )

    def as_spice_boolean_function(
        self,
        prop: Property,
        invert: bool = False,
    ) -> spiceypy.utils.callbacks.SpiceUDFUNB:
        """Wrap *prop* as a SPICE ``UDFUNB`` boolean callback."""
        if invert:
            def _fn(udfun: Any, t: float) -> bool:
                return not self.evaluate_scalar_raw(prop, float(t))
        else:
            def _fn(udfun: Any, t: float) -> bool:
                return bool(self.evaluate_scalar_raw(prop, float(t)))

        return spiceypy.utils.callbacks.SpiceUDFUNB(_fn)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_evaluator: Evaluator | None = None


def get_evaluator() -> Evaluator:
    """Return the global :class:`Evaluator` singleton, initialising it on first call.

    Initialisation:

    1. Creates a fresh :class:`~spice_segmenter.engines.spice_engine.SpiceEngine`.
    2. Calls :func:`~spice_segmenter.computations.spice.register_all` to populate
       it with all built-in SPICE compute functions.
    3. Wraps it in an :class:`Evaluator` and caches the result.

    All imports are deferred to avoid circular-import issues at module load time.
    """
    global _evaluator
    if _evaluator is None:
        from .spice_engine import SpiceEngine
        from ..computations.spice import register_all

        engine = SpiceEngine()
        register_all(engine)
        _evaluator = Evaluator(engine)
    return _evaluator
