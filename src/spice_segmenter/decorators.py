from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pint
from attr import define

if TYPE_CHECKING:
    from spice_segmenter.trajectory_properties import PropertyTypes


def vectorize(
    function: Optional[Callable[..., Any]] = None,
    otypes: Optional[str] = None,
    signature: Optional[str] = None,
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
    """Declare a property.

    Utility decorator to speed-up building new properties.

    In a future this might also register the property in a global registry.

    Parameters
    ----------

    name : str
        Name of the property

    unit : pint.Unit
        Unit of the property

    property_type : PropertyTypes
        Type of the property

    """

    if cls is None:
        return partial(declare, name=name, unit=unit, property_type=property_type)

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

    return define(repr=False, order=False, eq=False)(cls)
