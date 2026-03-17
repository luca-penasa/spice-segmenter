"""Tests for @property_function decorator."""

import pytest
import pint
from spice_segmenter.core.property_decorator import property_function, _to_class_name
from spice_segmenter.core.property import Property, PropertyTypes
from spice_segmenter.core.constraints import Constraint


def test_to_class_name():
    """Test snake_case to PascalCase conversion."""
    assert _to_class_name("distance") == "Distance"
    assert _to_class_name("phase_angle") == "PhaseAngle"
    assert _to_class_name("angular_size") == "AngularSize"
    assert _to_class_name("sub_observer_point") == "SubObserverPoint"


def test_property_function_basic():
    """Test basic property function generation."""
    
    @property_function(name="test_prop", unit="km")
    def test_property(time, observer, target):
        return 1000.0
    
    # Check it's a class
    assert isinstance(test_property, type)
    assert issubclass(test_property, Property)
    
    # Check metadata
    assert test_property._name == "test_prop"
    assert str(test_property._unit) == "kilometer"
    assert test_property._type == PropertyTypes.SCALAR


def test_property_function_with_defaults():
    """Test property function with default parameters."""
    
    @property_function(name="test_prop2", unit="rad")
    def test_property(time, observer, target, correction="NONE"):
        return 0.5 if correction == "LT" else 0.0
    
    # Create instances with and without default
    prop1 = test_property("JUICE", "GANYMEDE")
    prop2 = test_property("JUICE", "GANYMEDE", correction="LT")
    
    # Check they behave differently
    assert prop1(0.0) == 0.0
    assert prop2(0.0) == 0.5


def test_property_function_instantiation():
    """Test that generated property can be instantiated."""
    
    @property_function(name="test_dist", unit="km")
    def test_distance(time, observer, target):
        return 5000.0
    
    prop = test_distance("JUICE", "GANYMEDE")
    
    assert hasattr(prop, 'observer')
    assert hasattr(prop, 'target')
    assert prop.observer == "JUICE"
    assert prop.target == "GANYMEDE"


def test_property_function_call():
    """Test calling generated property."""
    
    @property_function(name="test_call", unit="km")
    def test_property(time, observer, target):
        return time * 10.0  # Simple test function
    
    prop = test_property("JUICE", "GANYMEDE")
    result = prop(100.0)
    
    assert result == 1000.0


def test_property_function_constraint_creation():
    """Test that generated properties support constraint operators."""
    
    @property_function(name="test_constraint", unit="km")
    def test_distance(time, observer, target):
        return 5000.0
    
    prop = test_distance("JUICE", "GANYMEDE")
    constraint = prop < "10000 km"
    
    assert isinstance(constraint, Constraint)
    assert constraint.operator == "<"


def test_property_function_metadata_storage():
    """Test that metadata is stored on the class."""
    
    @property_function(name="test_meta", unit="deg", property_type=PropertyTypes.SCALAR)
    def test_property(time, observer, target):
        return 45.0
    
    assert hasattr(test_property, '_metadata')
    assert hasattr(test_property, '_compute_fn')
    
    metadata = test_property._metadata
    assert metadata.name == "test_meta"
    assert metadata.property_type == PropertyTypes.SCALAR
    assert len(metadata.parameter_names) == 2  # observer, target


def test_property_function_docstring():
    """Test that docstring is preserved."""
    
    @property_function(name="test_doc", unit="km")
    def test_property(time, observer, target):
        """This is a test property."""
        return 1000.0
    
    assert test_property.__doc__ is not None
    assert "test property" in test_property.__doc__.lower()


def test_property_function_vector_unit():
    """Test property with vector units."""
    
    @property_function(
        name="test_vector",
        unit=["km", "km", "km"],
        property_type=PropertyTypes.VECTOR
    )
    def test_vector(time, observer, target):
        return [1.0, 2.0, 3.0]
    
    assert test_vector._type == PropertyTypes.VECTOR
    assert isinstance(test_vector._unit, list)
    assert len(test_vector._unit) == 3


def test_property_function_boolean_type():
    """Test creating boolean property."""
    
    @property_function(
        name="test_bool",
        unit="dimensionless",
        property_type=PropertyTypes.BOOLEAN
    )
    def test_bool(time, observer, target):
        return True
    
    assert test_bool._type == PropertyTypes.BOOLEAN
    
    prop = test_bool("JUICE", "GANYMEDE")
    assert prop(0.0) == True


def test_property_function_no_observer_target():
    """Test property without observer/target fields."""
    
    @property_function(name="test_custom", unit="m")
    def test_property(time, foo, bar):
        return foo + bar
    
    prop = test_property(10, 20)
    assert hasattr(prop, 'foo')
    assert hasattr(prop, 'bar')
    assert prop.foo == 10
    assert prop.bar == 20


def test_property_function_with_type_annotations():
    """Test that type annotations are extracted."""
    
    @property_function(name="test_typed", unit="km")
    def test_property(time, observer: str, target: str, value: float = 100.0):
        return value
    
    metadata = test_property._metadata
    assert 'observer' in metadata.parameter_types
    assert 'target' in metadata.parameter_types
    assert 'value' in metadata.parameter_types
    assert 'value' in metadata.parameter_defaults
    assert metadata.parameter_defaults['value'] == 100.0


def test_property_function_class_name():
    """Test that class gets proper name."""
    
    @property_function(name="test_angular_separation_cls", unit="rad")
    def test_angular_sep(time, observer, target):
        return 0.5
    
    assert test_angular_sep.__name__ == "TestAngularSeparationCls"
    assert test_angular_sep.__qualname__ == "TestAngularSeparationCls"
