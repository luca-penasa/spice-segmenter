"""Base interface for computation engines."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ComputationEngine(Protocol):
    """Protocol for computation engines.

    An engine stores scalar and (optionally) vector compute functions keyed by
    the exact property *class* object.  Registration is class-keyed — not
    string-keyed — so that:

    * A missing import causes an immediate ``ImportError`` rather than a
      silent runtime failure.
    * Renaming ``_name`` for display purposes never breaks dispatch.
    * The engine can walk the MRO to handle inheritance transparently.
    """

    def register(
        self,
        property_class: type,
        *,
        scalar_fn: Callable | None = None,
        vector_fn: Callable | None = None,
        priority: int = 0,
    ) -> None:
        """Register compute functions for *property_class*.

        At least one of *scalar_fn* or *vector_fn* must be provided.

        Args:
            property_class: The exact Property subclass to register for.
            scalar_fn: ``fn(prop, time_et: float) -> Any`` for single times.
            vector_fn: ``fn(prop, times_et: ndarray) -> ndarray`` for arrays.
            priority: Higher values are tried first when multiple functions
                are registered for the same class (e.g. by different backends).
        """
        ...

    def evaluate_scalar(self, prop: Any, time_et: float) -> Any:
        """Evaluate *prop* at a single pre-converted SPICE ET.

        Raises:
            KeyError: If no scalar function is registered for ``type(prop)``
                or any of its bases.
        """
        ...

    def evaluate_vector(self, prop: Any, times_et: np.ndarray) -> np.ndarray:
        """Evaluate *prop* at an array of pre-converted SPICE ETs.

        Falls back to ``np.vectorize(scalar_fn)`` when no dedicated vector
        function has been registered.

        Raises:
            KeyError: If neither a vector nor a scalar function is registered.
        """
        ...

    def can_evaluate(self, property_class: type) -> bool:
        """Return ``True`` if *property_class* (or a base) has a scalar fn."""
        ...
