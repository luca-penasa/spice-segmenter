from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import pint
from attr import define

if TYPE_CHECKING:
    from ..core.property import PropertyTypes


class PropertyMeta(type):
    """Metaclass for Property classes that auto-generates metadata properties.
    
    This metaclass automatically creates property accessors for _name and _unit
    class attributes, and registers properties in a global registry.
    
    Usage:
        class MyProperty(Property, metaclass=PropertyMeta):
            _name = "my_property"
            _unit = pint.Unit("km")
    """
    
    # Class-level registry
    registry: dict[str, type] = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # Extract metadata before creating class
        _name = namespace.get('_name', '')
        _unit = namespace.get('_unit', pint.Unit('dimensionless'))
        _type = namespace.get('_type', None)
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # If this class defines _name, set up property accessors and register
        if _name:
            # Create name property
            cls.name = property(lambda self: _name)
            # Register in global registry
            mcs.registry[_name] = cls
        
        # Create unit property if _unit is defined
        if _unit is not None:
            cls.unit = property(lambda self: _unit)
        
        # Create type property if _type is specified
        if _type:
            cls.type = property(lambda self: _type)
        
        return cls


def vectorize(
    function: Callable[..., Any] | None = None,
    otypes: str | None = None,
    signature: str | None = None,
) -> Callable[..., Any]:
    """Numpy vectorization wrapper that works with instance methods."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args) -> Any:  # type: ignore
            return vectorized(*args)

        return wrapper

    if function:
        return decorator(function)

    return decorator


def declare(
    cls=None,
    name: str = "",
    unit: pint.Unit = pint.Unit("dimensionless"),
    property_type: "PropertyTypes|None" = None,
):
    """DEPRECATED: Use PropertyMeta metaclass instead.
    
    Legacy decorator for backward compatibility. Properties should now use:
    
        class MyProperty(Property, metaclass=PropertyMeta):
            _name = "my_property"
            _unit = pint.Unit("km")
    
    This decorator is kept for backward compatibility and works by:
    1. Setting name, unit, and type properties on the class
    2. Applying the @define decorator
    3. Registering in PropertyMeta.registry
    """
    if cls is None:
        from functools import partial
        return partial(declare, name=name, unit=unit, property_type=property_type)

    # Set the properties BEFORE applying @define
    def _name(self):
        return name
    
    def _unit(self):
        return unit
    
    cls.name = property(_name)
    cls.unit = property(_unit)
    
    if property_type:
        def _type(self):
            return property_type
        cls.type = property(_type)
    
    # Now apply attrs define
    P = define(repr=False, order=False, eq=False)(cls)
    
    # Register in the global registry
    PropertyMeta.registry[name] = P
    
    return P
