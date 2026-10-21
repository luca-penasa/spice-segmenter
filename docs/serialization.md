# Serialization Module

The `spice_segmenter.serialization` module provides utilities for serializing and deserializing `Property` and `Constraint` objects to/from dictionaries and JSON.

## Quick Start

```python
from spice_segmenter import (
    PhaseAngle, 
    Distance,
    create_property_converter,
    unstructure_constraint,
    structure_constraint
)

# Create some constraints
c1 = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
c2 = Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
cc = c1 & c2

# Option 1: Using the to_json/from_json methods (simplest)
json_str = cc.to_json(indent=2)
reconstructed = Constraint.from_json(json_str)

# Option 2: Using convenience functions
data = unstructure_constraint(cc)
reconstructed = structure_constraint(data)

# Option 3: Create a converter for multiple operations
converter = create_property_converter()
data = converter.unstructure(cc)
reconstructed = converter.structure(data, Constraint)
```

## Features

- **Tagged Union Serialization**: Uses class names as type discriminators
- **Automatic Type Detection**: Reconstructs the correct class hierarchy
- **JSON Compatible**: All serialized data can be converted to JSON
- **External Type Support**: Handles `pint.Quantity`, `SpiceBody`, and `SpiceInstrument`

## API Reference

### Property Methods

#### `property.to_json(indent=None)`

Serialize a Property or Constraint to a JSON string.

**Parameters:**
- `indent` (int, optional): Number of spaces for indentation. None for compact format.

**Returns:** JSON string representation

**Example:**
```python
c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"
json_str = c.to_json(indent=2)
```

#### `Property.from_json(json_str)`

Deserialize a Property or Constraint from a JSON string.

**Parameters:**
- `json_str` (str): JSON string representation

**Returns:** Reconstructed Property object

**Example:**
```python
reconstructed = Constraint.from_json(json_str)
```

### Module Functions

### `create_property_converter(register_spice_types=True, register_pint=True)`

Creates a converter for Property and Constraint objects, with `unstructure(obj)` and
`structure(data, cls)` methods. SPICE references and pint units/quantities are serialized as strings.

**Parameters:**
- `register_spice_types`, `register_pint`: accepted for compatibility, no effect

**Returns:** `PropertyConverter` instance

### `unstructure_constraint(constraint, converter=None)`

Serialize a Constraint to a dictionary.

**Parameters:**
- `constraint` (Constraint): The constraint to serialize
- `converter` (Converter, optional): Pre-configured converter

**Returns:** Dictionary representation

### `structure_constraint(data, converter=None)`

Deserialize a dictionary into a Constraint.

**Parameters:**
- `data` (dict): Dictionary representation
- `converter` (Converter, optional): Pre-configured converter

**Returns:** Reconstructed Constraint object

## Example: JSON Serialization

The simplest way to serialize to JSON is using the `to_json()` and `from_json()` methods:

```python
from spice_segmenter import PhaseAngle

# Create constraint
c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

# Serialize to JSON - super simple!
json_str = c.to_json(indent=2)
print(json_str)

# Deserialize from JSON
from spice_segmenter import Constraint
reconstructed = Constraint.from_json(json_str)
```

You can also use the lower-level functions for more control:

```python
import json
from spice_segmenter import PhaseAngle, unstructure_constraint, structure_constraint

# Create constraint
c = PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg"

# Serialize to JSON
data = unstructure_constraint(c)
json_str = json.dumps(data, indent=2)

# Deserialize from JSON
loaded_data = json.loads(json_str)
reconstructed = structure_constraint(loaded_data)
```

## Example: Reusing Converter

```python
from spice_segmenter import create_property_converter, PhaseAngle, Distance

# Create converter once
converter = create_property_converter()

# Use for multiple constraints
constraints = [
    PhaseAngle("JUICE_JANUS", "GANYMEDE") > "20 deg",
    Distance("JUICE_JANUS", "GANYMEDE") < "5e4 km"
]

# Serialize all
serialized = [converter.unstructure(c) for c in constraints]

# Deserialize all
from spice_segmenter import Constraint
reconstructed = [converter.structure(data, Constraint) for data in serialized]
```

## Serialized Format

The serialized format uses tagged unions with the `type` field as discriminator:

```json
{
  "time_step": null,
  "left": {
    "observer": "JUICE_JANUS",
    "target": "GANYMEDE",
    "light_time_correction": "NONE",
    "third_body": "SUN",
    "type": "PhaseAngle"
  },
  "right": {
    "_value": "20 degree",
    "type": "ScalarConstant"
  },
  "operator": ">",
  "type": "Constraint"
}
```

Boolean constants are written as JSON booleans (`"_value": true`); the former `"True"` / `"False"`
strings are still read.

## JSON Schema

`spice_segmenter.support.serialization.json_schema()` returns the JSON Schema (draft 2020-12) of the
format: one definition per class (its fields, with their constraints, plus a constant `type`), and
`#/$defs/Node` for any of them. It is built from the classes currently imported, so properties
registered by plug-in modules are included once their module is loaded.

```python
import jsonschema
from spice_segmenter.support.serialization import json_schema

jsonschema.validate(unstructure_constraint(c), json_schema())
```

## Technical Details

Properties and constraints are pydantic dataclasses (`spice_segmenter.support.model.segmenter_dataclass`):

1. each node is dumped by its pydantic serializer and tagged with its class name under `type`;
2. nested properties / constraints (`Constraint.left` / `right`, `Inverted.parent`,
   `ComponentSelector.vector`, ...) are fields of type `PropertyNode`, which serialize and validate
   through the same tagged form, so a whole tree round-trips;
3. SPICE references, pint units and quantities are annotated types
   (`spice_segmenter.support.types`) that convert their input and serialize to strings;
4. on loading, the `type` tag selects the class among the `Property` subclasses and the remaining
   keys are passed to its constructor (validated by pydantic).
