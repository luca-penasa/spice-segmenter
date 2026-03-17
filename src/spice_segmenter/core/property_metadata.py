"""Property metadata extracted from function definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import pint

from .property import PropertyTypes


@dataclass
class PropertyMetadata:
    """Metadata extracted from a property function.
    
    This encapsulates all information needed to generate a Property class
    from a function definition, including parameter types, defaults, and
    computation backends.
    """
    
    name: str
    """Property identifier (e.g., 'distance', 'phase_angle')"""
    
    unit: pint.Unit | list[pint.Unit]
    """Property unit(s)"""
    
    property_type: PropertyTypes
    """Property type (SCALAR, BOOLEAN, VECTOR, DISCRETE)"""
    
    compute_fn: Callable
    """Function that computes the property value"""
    
    parameter_names: list[str]
    """Names of function parameters (excluding 'time')"""
    
    parameter_types: dict[str, type] = field(default_factory=dict)
    """Type annotations for parameters"""
    
    parameter_defaults: dict[str, Any] = field(default_factory=dict)
    """Default values for optional parameters"""
    
    backend_hints: dict[str, Any] = field(default_factory=dict)
    """Backend-specific metadata"""
    
    vectorized: bool = True
    """Whether to auto-vectorize the function"""
    
    doc: str | None = None
    """Function docstring"""
    
    def get_parameter_signature(self) -> str:
        """Get parameter signature as string for debugging."""
        params = []
        for name in self.parameter_names:
            param_str = name
            if name in self.parameter_types:
                param_str += f": {self.parameter_types[name].__name__}"
            if name in self.parameter_defaults:
                param_str += f" = {repr(self.parameter_defaults[name])}"
            params.append(param_str)
        return f"({', '.join(params)})"
