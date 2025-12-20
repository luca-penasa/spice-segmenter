from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import pint

if TYPE_CHECKING:
    from ..core.property import Property, PropertyTypes


# Global property registry
# Maps property name → Property class
PROPERTY_REGISTRY: dict[str, type["Property"]] = {}


def register_property(name: str, property_class: type["Property"]) -> type["Property"]:
    """Register a property class in the global registry.
    
    Args:
        name: Property name (e.g., "distance", "phase_angle")
        property_class: The Property subclass to register
        
    Returns:
        The property_class (allows use as a decorator)
        
    Raises:
        TypeError: If name is not a string
        
    Example:
        >>> @register_property("distance")
        ... class Distance(Property):
        ...     _name = "distance"
        ...     _unit = pint.Unit("km")
    """
    if not isinstance(name, str):
        # Silently ignore non-string keys (likely descriptors)
        return property_class
    
    PROPERTY_REGISTRY[name] = property_class
    return property_class


def get_property_class(name: str) -> type["Property"] | None:
    """Get a property class by name from the registry.
    
    Args:
        name: Property name to look up
        
    Returns:
        Property class if found, None otherwise
    """
    return PROPERTY_REGISTRY.get(name)


def list_registered_properties() -> dict[str, type["Property"]]:
    """Get all registered properties.
    
    Returns:
        Dictionary mapping property names to classes
    """
    return PROPERTY_REGISTRY.copy()


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

