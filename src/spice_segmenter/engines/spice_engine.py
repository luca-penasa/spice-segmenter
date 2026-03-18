"""SpiceEngine: class-keyed registry of scalar / vector compute functions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SpiceEngine:
    """Default computation engine backed by direct SPICE toolkit calls.

    Compute functions are registered per property *class* (not string name),
    which means:

    * An ``ImportError`` is raised immediately if the class doesn't exist.
    * Renaming ``_name`` for display never silently breaks dispatch.
    * MRO walk allows subclasses to inherit a parent's implementation.

    Usage::

        engine = SpiceEngine()
        engine.register(Distance, scalar_fn=distance_scalar, vector_fn=distance_vector)
        engine.register(PhaseAngle, scalar_fn=phase_angle_scalar)  # vector auto-made

    Registration is order-independent; entries are sorted by *priority* (highest
    first) so that a higher-priority override can replace the default.
    """

    def __init__(self) -> None:
        # dict[type, list[tuple[int, Callable]]] — sorted descending by priority
        self._scalar_fns: dict[type, list[tuple[int, Callable]]] = {}
        self._vector_fns: dict[type, list[tuple[int, Callable]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        property_class: type,
        *,
        scalar_fn: Callable | None = None,
        vector_fn: Callable | None = None,
        priority: int = 0,
    ) -> None:
        """Register scalar and/or vector compute functions for *property_class*.

        At least one of *scalar_fn* / *vector_fn* must be supplied.  Both may
        be provided when a C-level vectorised variant exists.
        """
        if scalar_fn is None and vector_fn is None:
            raise ValueError(
                f"register({property_class.__name__}): "
                "at least one of scalar_fn or vector_fn must be provided."
            )
        if scalar_fn is not None:
            bucket = self._scalar_fns.setdefault(property_class, [])
            bucket.append((priority, scalar_fn))
            bucket.sort(key=lambda x: x[0], reverse=True)
        if vector_fn is not None:
            bucket = self._vector_fns.setdefault(property_class, [])
            bucket.append((priority, vector_fn))
            bucket.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Lookup helpers (MRO walk)
    # ------------------------------------------------------------------

    def _lookup_scalar(self, prop_type: type) -> Callable | None:
        """Return highest-priority scalar fn for *prop_type* via MRO walk."""
        for cls in prop_type.__mro__:
            if cls in self._scalar_fns:
                return self._scalar_fns[cls][0][1]
        return None

    def _lookup_vector(self, prop_type: type) -> Callable | None:
        """Return highest-priority vector fn for *prop_type* via MRO walk."""
        for cls in prop_type.__mro__:
            if cls in self._vector_fns:
                return self._vector_fns[cls][0][1]
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_scalar(self, prop: Any, time_et: float) -> Any:
        """Evaluate *prop* at a single pre-converted SPICE ET (float seconds).

        Raises:
            KeyError: when no scalar fn is registered for ``type(prop)`` or any base.
        """
        fn = self._lookup_scalar(type(prop))
        if fn is None:
            raise KeyError(type(prop))
        return fn(prop, time_et)

    def evaluate_vector(self, prop: Any, times_et: np.ndarray) -> np.ndarray:
        """Evaluate *prop* at an array of SPICE ETs.

        If no dedicated vector fn is registered, falls back to
        ``np.vectorize(scalar_fn)`` using ``prop._vector_output_shape`` as the
        numpy signature (``None`` = each call returns a scalar).

        Raises:
            KeyError: when neither a vector nor a scalar fn is registered.
        """
        vector_fn = self._lookup_vector(type(prop))
        if vector_fn is not None:
            return vector_fn(prop, times_et)

        # No dedicated vector fn — fall back to scalar if available
        scalar_fn = self._lookup_scalar(type(prop))
        if scalar_fn is None:
            raise KeyError(type(prop))
        sig = getattr(prop, "_vector_output_shape", None)
        return np.vectorize(
            lambda t: scalar_fn(prop, t), signature=sig
        )(times_et)

    def can_evaluate(self, property_class: type) -> bool:
        """Return ``True`` if *property_class* (or any MRO base) has a scalar fn."""
        for cls in property_class.__mro__:
            if cls in self._scalar_fns:
                return True
        return False
