"""Decorator for defining properties from functions."""

from __future__ import annotations
from typing import Callable, Any, TypeVar
import inspect
import pint

from .property import Property, PropertyTypes
from .property_metadata import PropertyMetadata
from ..support.decorators import vectorize, register_property

F = TypeVar('F', bound=Callable[..., Any])


def property_function(
    name: str,
    unit: str | pint.Unit | list[str | pint.Unit],
    property_type: PropertyTypes = PropertyTypes.SCALAR,
    vectorized: bool = True,
    backend: str | None = None,
    **backend_hints
) -> Callable[[F], type[Property]]:
    """
    Decorator to define a property from a function.
    
    Generates a Property class that wraps the function and supports
    backend dispatch, operator overloading, and constraint creation.
    
    The function signature (excluding 'time') defines the property's fields.
    Type annotations and defaults are preserved in the generated class.
    
    Args:
        name: Property identifier (e.g., "distance", "phase_angle")
        unit: Property unit(s) - string or pint.Unit or list
        property_type: SCALAR, BOOLEAN, VECTOR, or DISCRETE
        vectorized: Whether to auto-vectorize the function (default True)
        backend: Preferred backend name (None = auto-detect)
        **backend_hints: Additional metadata for backends
    
    Returns:
        Decorator that creates a Property class
    
    Example:
        >>> @property_function(
        ...     name="distance",
        ...     unit="km",
        ...     property_type=PropertyTypes.SCALAR
        ... )
        ... def compute_distance(
        ...     time: TIMES_TYPES,
        ...     observer: SpiceInstrument,
        ...     target: SpiceBody,
        ...     light_time_correction: str = "NONE"
        ... ) -> float:
        ...     return spiceypy.vnorm(
        ...         spiceypy.spkpos(
        ...             target.name, et(time),
        ...             observer.frame.name,
        ...             light_time_correction,
        ...             observer.name
        ...         )[0]
        ...     )
        >>> 
        >>> # Generated class usage (same as before):
        >>> dist = compute_distance("JUICE_JANUS", "GANYMEDE")
        >>> constraint = dist < "5e4 km"
    """
    
    def decorator(fn: F) -> type[Property]:
        # Extract function signature
        sig = inspect.signature(fn)
        param_names = []
        param_types = {}
        param_defaults = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'time':
                continue  # Skip time parameter
            
            param_names.append(param_name)
            
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = param.annotation
            
            if param.default != inspect.Parameter.empty:
                param_defaults[param_name] = param.default
        
        # Convert units to pint.Unit
        if isinstance(unit, list):
            units = [pint.Unit(u) if isinstance(u, str) else u for u in unit]
        else:
            units = pint.Unit(unit) if isinstance(unit, str) else unit
        
        # Create metadata
        metadata = PropertyMetadata(
            name=name,
            unit=units,
            property_type=property_type,
            compute_fn=fn,
            parameter_names=param_names,
            parameter_types=param_types,
            parameter_defaults=param_defaults,
            backend_hints=backend_hints,
            vectorized=vectorized,
            doc=fn.__doc__
        )
        
        # Generate Property class
        property_class = _generate_property_class(metadata, backend)
        
        # Store metadata on class for introspection
        property_class._compute_fn = fn
        property_class._metadata = metadata
        
        # Register in global property registry
        register_property(name, property_class)
        
        return property_class
    
    return decorator


def _generate_property_class(
    metadata: PropertyMetadata,
    backend_name: str | None
) -> type[Property]:
    """Generate a Property class from metadata.
    
    This function dynamically creates a Property subclass with:
    - Appropriate base class (Property or TargetedProperty)
    - attrs fields from function parameters
    - __call__ method with backend dispatch
    - Metadata properties (_name, _unit, _type)
    
    Args:
        metadata: Property metadata from function
        backend_name: Preferred backend (None = auto-select)
        
    Returns:
        Generated Property class
    """
    from attrs import define, field
    from ..support.spice_utilities import as_spice_ref
    from ..backends import get_backend
    
    # Determine base class
    has_observer_target = (
        'observer' in metadata.parameter_names 
        and 'target' in metadata.parameter_names
    )
    
    if has_observer_target:
        from ..properties.observation_properties import TargetedProperty
        base_class = TargetedProperty
    else:
        base_class = Property
    
    # Build attrs fields dictionary
    fields_dict = {}
    for param_name in metadata.parameter_names:
        field_kwargs = {}
        
        # Add converter for SPICE types
        if param_name in ('observer', 'target', 'third_body', 'reflector', 
                          'light_source', 'other'):
            field_kwargs['converter'] = as_spice_ref
        
        # Add default if present
        if param_name in metadata.parameter_defaults:
            field_kwargs['default'] = metadata.parameter_defaults[param_name]
        
        fields_dict[param_name] = field(**field_kwargs)
    
    # Create __call__ method with backend dispatch
    def __call__(self, time):
        """Compute property value with backend dispatch."""
        # Try backend dispatch first
        # Pass time as hint for backends to check if vectorization is needed
        backend_kwargs = {'_time_hint': time}
        
        if backend_name:
            backend = get_backend(metadata.name, backend=backend_name, **backend_kwargs)
        else:
            backend = get_backend(metadata.name, **backend_kwargs)
        
        if backend is not None:
            # Collect parameters from self
            params = {
                name: getattr(self, name) 
                for name in metadata.parameter_names
            }
            return backend.compute(metadata.name, time, **params)
        
        # Fallback to direct function call
        params = {
            name: getattr(self, name)
            for name in metadata.parameter_names
        }
        return metadata.compute_fn(time, **params)
    
    # Vectorize if requested
    if metadata.vectorized:
        __call__ = vectorize(__call__)
    
    # Convert property name to class name (e.g., phase_angle -> PhaseAngle)
    cls_name = _to_class_name(metadata.name)
    
    # Create class dynamically
    if has_observer_target:
        # TargetedProperty already has observer/target fields
        # Add only additional fields
        additional_fields = {
            k: v for k, v in fields_dict.items() 
            if k not in ('observer', 'target')
        }
        
        # Create namespace with __call__ and metadata
        namespace = {
            '__call__': __call__,
            '_name': metadata.name,
            '_unit': metadata.unit,
            '_type': metadata.property_type,
            '__doc__': metadata.doc or f"Property: {metadata.name}",
            '__module__': base_class.__module__,
            # Marker so Property.__init_subclass__ uses register_or_skip
            # instead of the strict register — generated classes are alternative
            # implementations and must not overwrite authoritative registrations.
            '_generated_by_property_function': True,
        }
        
        # Add additional field definitions
        namespace.update(additional_fields)
        
        # Create class with type()
        GeneratedProperty = type(cls_name, (base_class,), namespace)
        
        # Apply @define if there are additional fields
        if additional_fields:
            GeneratedProperty = define(repr=False, order=False, eq=False)(GeneratedProperty)
    else:
        # Create new class with all fields
        namespace = {
            '__call__': __call__,
            '_name': metadata.name,
            '_unit': metadata.unit,
            '_type': metadata.property_type,
            '__doc__': metadata.doc or f"Property: {metadata.name}",
            '__module__': base_class.__module__,
            # Marker so Property.__init_subclass__ uses register_or_skip.
            '_generated_by_property_function': True,
        }
        
        # Add all field definitions
        namespace.update(fields_dict)
        
        # Create class with type()
        GeneratedProperty = type(cls_name, (base_class,), namespace)
        
        # Apply @define
        GeneratedProperty = define(repr=False, order=False, eq=False)(GeneratedProperty)
    
    return GeneratedProperty


def _to_class_name(property_name: str) -> str:
    """Convert property_name to ClassName (e.g., phase_angle -> PhaseAngle).
    
    Args:
        property_name: Snake_case property name
        
    Returns:
        PascalCase class name
    """
    return ''.join(word.capitalize() for word in property_name.split('_'))
