from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.property import Property


# ---------------------------------------------------------------------------
# Registry shims — thin wrappers kept for backwards compatibility.
# The single source of truth is spice_segmenter.core.registry.
# ---------------------------------------------------------------------------


def get_property_class(name: str) -> type["Property"] | None:
    """Return the property class registered under *name*, or ``None``."""
    from spice_segmenter.core import registry
    return registry.property_registry.get(name)


def list_registered_properties() -> dict[str, type["Property"]]:
    """Return a snapshot copy of the full property registry."""
    from spice_segmenter.core.registry import all as _all
    return _all()


def vectorize(
    function: Callable[..., Any] | None = None,
    otypes: str | None = None,
    signature: str | None = None,
) -> Callable[..., Any]:
    """Numpy vectorization wrapper that works with instance methods.
    
    Wraps numpy.vectorize to handle instance methods properly by preserving
    function metadata via functools.wraps.
    
    Args:
        function: Function to vectorize (optional, allows use as @vectorize())
        otypes: Output types for numpy.vectorize
        signature: Signature for numpy.vectorize (e.g., "(),()->()")
        
    Returns:
        Vectorized function
        
    Example:
        >>> @vectorize(signature="(),()->()")
        ... def __call__(self, time: TIMES_TYPES) -> float:
        ...     return spiceypy.vnorm(spiceypy.spkpos(...))
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args) -> Any:  # type: ignore
            return vectorized(*args)

        return wrapper

    if function:
        return decorator(function)

    return decorator

