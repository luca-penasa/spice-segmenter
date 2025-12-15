"""
Serialization module for spice_segmenter Properties and Constraints.

This module provides utilities for serializing and deserializing Property and Constraint
objects using cattrs with tagged unions based on class names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_type_hints

import pint
from cattrs.preconf.json import make_converter
from cattrs.strategies import configure_tagged_union, include_subclasses

if TYPE_CHECKING:
    from cattrs import Converter

from spice_segmenter.constraint import Constraint
from spice_segmenter.property_base import Property


def get_all_subclasses(cls: type) -> list[type]:
    """
    Recursively get all subclasses of a class.
    
    Args:
        cls: The base class to get subclasses for
        
    Returns:
        List of all subclasses (including nested subclasses)
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def create_property_converter(
    register_spice_types: bool = True,
    register_pint: bool = True,
) -> Converter:
    """
    Create a cattrs converter configured for spice_segmenter Property and Constraint objects.
    
    This converter handles serialization and deserialization of Property hierarchies using
    tagged unions with the class name as the discriminator field ('type').
    
    Args:
        register_spice_types: If True, registers unstructure hooks for SpiceBody and 
                             SpiceInstrument to convert them to strings
        register_pint: If True, registers unstructure hook for pint.Quantity to convert
                      to string representation
    
    Returns:
        Configured cattrs Converter instance
        
    Example:
        >>> from spice_segmenter.serialization import create_property_converter
        >>> from spice_segmenter import PhaseAngle, Distance
        >>> 
        >>> # Create constraints
        >>> c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
        >>> c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
        >>> cc = c1 & c2
        >>> 
        >>> # Create converter and serialize
        >>> converter = create_property_converter()
        >>> data = converter.unstructure(cc)
        >>> 
        >>> # Deserialize back
        >>> from spice_segmenter import Constraint
        >>> reconstructed = converter.structure(data, Constraint)
    """
    converter = make_converter()

    # Register unstructuring hooks for external types
    if register_spice_types:
        from planetary_coverage.spice import (
            SpiceBody,
            SpiceFrame,
            SpiceInstrument,
            SpiceObserver,
            SpiceSpacecraft,
        )
        converter.register_unstructure_hook(SpiceInstrument, lambda x: str(x))
        converter.register_unstructure_hook(SpiceBody, lambda x: str(x))
        converter.register_unstructure_hook(SpiceFrame, lambda x: str(x))
        converter.register_unstructure_hook(SpiceObserver, lambda x: str(x))
        converter.register_unstructure_hook(SpiceSpacecraft, lambda x: str(x))

    if register_pint:
        converter.register_unstructure_hook(pint.Quantity, lambda x: str(x))
        converter.register_unstructure_hook(pint.Unit, lambda x: str(x))

    # Create a custom union strategy that uses "type" as the discriminator
    def custom_union_strategy(union, converter_arg):
        """Strategy that uses 'type' field as discriminator with class name as value"""
        configure_tagged_union(
            union,
            converter_arg,
            tag_name="type",
            tag_generator=lambda cls: cls.__name__,
        )

    # Apply include_subclasses for Property - this creates hooks for all Property subclasses
    # This handles the general case for most Property types
    include_subclasses(
        Property,
        converter,
        union_strategy=custom_union_strategy,
    )

    # Handle the specific Property | ConstraintBase union that appears in Constraint fields
    # This is necessary because cattrs sees this as a distinct union type

    # Build a registry of all Property subclasses for the custom structure function
    property_subclasses = get_all_subclasses(Property)
    class_registry = {cls.__name__: cls for cls in property_subclasses}

    def structure_property_or_constraint(obj, cl):
        """
        Structure function for Property | ConstraintBase unions.
        
        Uses the 'type' field to determine the correct class and delegates
        to the converter for actual structuring.
        """
        if isinstance(obj, dict) and "type" in obj:
            type_name = obj["type"]
            if type_name in class_registry:
                target_class = class_registry[type_name]
                return converter.structure(obj, target_class)
        raise ValueError(
            f"Cannot structure {obj} as Property | ConstraintBase. "
            f"Expected dict with 'type' field containing a known class name.",
        )

    # Register this custom structure hook for the Property | ConstraintBase union
    # that appears in Constraint.left and Constraint.right
    hints = get_type_hints(Constraint)
    prop_or_const_union = hints["left"]  # Both 'left' and 'right' have the same type
    converter.register_structure_hook(prop_or_const_union, structure_property_or_constraint)

    return converter


def unstructure_constraint(constraint: Constraint, converter: Converter | None = None) -> dict:
    """
    Convenience function to unstructure (serialize) a Constraint to a dictionary.
    
    Args:
        constraint: The Constraint object to serialize
        converter: Optional pre-configured converter. If None, creates a new one.
        
    Returns:
        Dictionary representation of the constraint
    """
    if converter is None:
        converter = create_property_converter()
    return converter.unstructure(constraint)


def structure_constraint(data: dict, converter: Converter | None = None) -> Constraint:
    """
    Convenience function to structure (deserialize) a dictionary into a Constraint.
    
    Args:
        data: Dictionary representation of a constraint
        converter: Optional pre-configured converter. If None, creates a new one.
        
    Returns:
        Reconstructed Constraint object
    """
    if converter is None:
        converter = create_property_converter()
    return converter.structure(data, Constraint)
