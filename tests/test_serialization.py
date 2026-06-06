"""Tests for the serialization module."""


from spice_segmenter import (
    Constraint,
    Distance,
    PhaseAngle,
    create_property_converter,
    structure_constraint,
    unstructure_constraint,
)


def test_create_property_converter():
    """Test that create_property_converter creates a working converter."""
    converter = create_property_converter()
    assert converter is not None


def test_unstructure_simple_constraint():
    """Test unstructuring a simple constraint."""
    c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

    converter = create_property_converter()
    data = converter.unstructure(c)

    assert isinstance(data, dict)
    assert data["type"] == "Constraint"
    assert data["operator"] == ">"
    assert data["left"]["type"] == "PhaseAngle"
    assert data["right"]["type"] == "ScalarConstant"


def test_structure_simple_constraint():
    """Test structuring a simple constraint."""
    c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

    converter = create_property_converter()
    data = converter.unstructure(c)
    reconstructed = converter.structure(data, Constraint)

    assert type(reconstructed) == type(c)
    assert reconstructed.operator == c.operator


def test_unstructure_complex_constraint():
    """Test unstructuring a complex nested constraint."""
    c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
    c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
    cc = c1 & c2

    converter = create_property_converter()
    data = converter.unstructure(cc)

    assert isinstance(data, dict)
    assert data["type"] == "Constraint"
    assert data["operator"] == "&"
    assert data["left"]["type"] == "Constraint"
    assert data["right"]["type"] == "Constraint"


def test_structure_complex_constraint():
    """Test structuring a complex nested constraint."""
    c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
    c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
    cc = c1 & c2

    converter = create_property_converter()
    data = converter.unstructure(cc)
    reconstructed = converter.structure(data, Constraint)

    assert type(reconstructed) == type(cc)
    assert type(reconstructed.left) == type(cc.left)
    assert type(reconstructed.right) == type(cc.right)
    assert reconstructed.operator == cc.operator


def test_convenience_functions():
    """Test the convenience functions work correctly."""
    c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

    # Test unstructure convenience function
    data = unstructure_constraint(c)
    assert isinstance(data, dict)
    assert data["type"] == "Constraint"

    # Test structure convenience function
    reconstructed = structure_constraint(data)
    assert type(reconstructed) == type(c)


def test_roundtrip():
    """Test that serialize + deserialize produces equivalent object."""
    c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
    c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
    cc = c1 & c2

    data = unstructure_constraint(cc)
    reconstructed = structure_constraint(data)

    # Verify the structure is preserved
    assert type(reconstructed) == type(cc)
    assert type(reconstructed.left) == type(cc.left)
    assert type(reconstructed.right) == type(cc.right)
    assert type(reconstructed.left.left) == type(cc.left.left)
    assert type(reconstructed.left.right) == type(cc.left.right)


def test_property_to_json():
    """Test Property.to_json() method."""
    c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

    # Test compact format
    json_str = c.to_json()
    assert isinstance(json_str, str)
    assert "PhaseAngle" in json_str
    assert "GANYMEDE" in json_str

    # Test indented format
    json_str_indented = c.to_json(indent=2)
    assert isinstance(json_str_indented, str)
    assert len(json_str_indented) > len(json_str)  # Indented should be longer


def test_property_from_json():
    """Test Property.from_json() method."""
    c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

    json_str = c.to_json()
    reconstructed = Constraint.from_json(json_str)

    assert type(reconstructed) == type(c)
    assert reconstructed.operator == c.operator


def test_json_roundtrip():
    """Test that to_json + from_json preserves object structure."""
    c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
    c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
    cc = c1 & c2

    # Serialize to JSON and back
    json_str = cc.to_json(indent=2)
    reconstructed = Constraint.from_json(json_str)

    # Verify the structure is preserved
    assert type(reconstructed) == type(cc)
    assert type(reconstructed.left) == type(cc.left)
    assert type(reconstructed.right) == type(cc.right)


def test_property_to_json_simple():
    """Test to_json on a simple property (not a constraint)."""
    prop = PhaseAngle("JUICE_JANUS", "GANYMEDE")

    json_str = prop.to_json()
    assert isinstance(json_str, str)
    assert "PhaseAngle" in json_str

    # Reconstruct
    reconstructed = PhaseAngle.from_json(json_str)
    assert type(reconstructed) == type(prop)
