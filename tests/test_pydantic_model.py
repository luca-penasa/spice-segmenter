"""Behaviour of the pydantic dataclass layer (support.model / fields / types / serialization)."""

import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pint
import pytest
from pydantic import ValidationError

from spice_segmenter import (
    Config,
    Constraint,
    Distance,
    Inverted,
    MinMaxConstraint,
    PhaseAngle,
    Property,
    SubObserverLatitudinal,
    TimeSegment,
    TimeSegmentsCollection,
    property_registry,
)
from spice_segmenter.ops.constant_values import BoolConstant, ScalarConstant
from spice_segmenter.properties.component_selector import ComponentSelector
from spice_segmenter.properties.coordinates import BoresightIntersects
from spice_segmenter.properties.observation_properties import MinMaxConditionTypes
from spice_segmenter.support.fields import MISSING, evolve, fields
from spice_segmenter.support.model import segmenter_dataclass
from spice_segmenter.support.serialization import (
    json_schema,
    structure_node,
    unstructure_node,
)


def test_wire_format_of_a_property():
    d = Distance("JUICE_JANUS", "GANYMEDE")
    assert d.to_json() == (
        '{"observer": "JUICE_JANUS", "target": "GANYMEDE", "light_time_correction": "NONE", '
        '"unit": "kilometer", "type": "Distance"}'
    )


def test_wire_format_of_a_constraint():
    c = Distance("JUICE_JANUS", "GANYMEDE") > "2 km"
    data = unstructure_node(c)
    assert data["type"] == "Constraint"
    assert data["time_step"] is None
    assert data["operator"] == ">"
    assert data["right"] == {"_value": "2 kilometer", "type": "ScalarConstant"}


@pytest.mark.parametrize(
    "constraint",
    [
        ~(
            (Distance("JUICE_JANUS", "GANYMEDE") < 1000)
            | (PhaseAngle("JUICE_JANUS", "GANYMEDE") > "1 rad")
        ),
        MinMaxConstraint(
            Distance("JUICE_JANUS", "GANYMEDE"), MinMaxConditionTypes.GLOBAL_MINIMUM
        ),
        SubObserverLatitudinal("JUICE_JANUS", "GANYMEDE").latitude.as_unit("rad")
        > "0.1 rad",
        Constraint(
            Distance("JUICE_JANUS", "GANYMEDE"),
            ScalarConstant("3 km"),
            ">",
            time_step="10 min",
        ),
    ],
    ids=["inverted", "minmax", "component_selector", "time_step"],
)
def test_json_roundtrip_of_nested_trees(constraint):
    data = json.loads(json.dumps(unstructure_node(constraint)))
    rebuilt = structure_node(data)
    assert type(rebuilt) is type(constraint)
    assert unstructure_node(rebuilt) == unstructure_node(constraint)


def test_time_step_is_converted_to_seconds():
    c = Constraint(
        Distance("JUICE_JANUS", "GANYMEDE"),
        ScalarConstant("3 km"),
        ">",
        time_step="10 min",
    )
    assert c.time_step == 600.0


def test_bool_constant_serializes_as_json_bool_and_reads_former_strings():
    assert unstructure_node(BoolConstant(False)) == {
        "_value": False,
        "type": "BoolConstant",
    }
    assert structure_node({"_value": "False", "type": "BoolConstant"}).value is False
    assert structure_node({"_value": "True", "type": "BoolConstant"}).value is True


def test_private_fields_accept_their_public_name():
    assert ScalarConstant(value="2 km").value == 2
    assert ScalarConstant("2 km").unit == pint.Unit("km")


def test_unknown_keyword_is_rejected():
    with pytest.raises(ValidationError):
        Distance("JUICE_JANUS", "GANYMEDE", bogus=1)


def test_constraint_sides_are_required():
    with pytest.raises(ValidationError):
        Constraint(Distance("JUICE_JANUS", "GANYMEDE"))


def test_assignment_is_converted():
    cfg = Config()
    cfg.solver_step = 60
    assert cfg.solver_step == pd.Timedelta(seconds=60)

    d = Distance("JUICE_JANUS", "GANYMEDE")
    d.unit = "m"
    assert d.unit == pint.Unit("m")


def test_as_unit_keeps_init_false_fields_out():
    b = BoresightIntersects("JUICE_JANUS", "GANYMEDE")
    assert b.as_unit("").intersection is not None


def test_evolve():
    d = Distance("JUICE_JANUS", "GANYMEDE")
    e = evolve(d, target="CALLISTO")
    assert e.target.name == "CALLISTO"
    assert e.observer.name == d.observer.name
    with pytest.raises(ValidationError):
        evolve(d, bogus=1)


def test_fields_helper():
    by_name = {f.name: f for f in fields(Distance)}
    assert by_name["observer"].has_factory
    assert by_name["unit"].kw_only
    assert by_name["unit"].default == pint.Unit("km")

    by_name = {f.name: f for f in fields(ScalarConstant)}
    assert by_name["_value"].init_name == "value"
    assert by_name["_value"].default is MISSING
    assert not by_name["_value"].has_default


def test_registry_holds_the_final_classes():
    assert property_registry["distance"] is Distance
    assert property_registry["component_selector"] is ComponentSelector


def test_subclass_registration_and_super():
    @segmenter_dataclass
    class _PydanticTestProperty(Distance):
        _name = "_pydantic_test_property"

        def __repr__(self) -> str:
            return "test " + super().__repr__()

    try:
        assert property_registry["_pydantic_test_property"] is _PydanticTestProperty
        p = _PydanticTestProperty("JUICE_JANUS", "GANYMEDE")
        assert repr(p).startswith("test Distance")
        assert isinstance(Property.from_json(p.to_json()), _PydanticTestProperty)
    finally:
        # keep the registry as other tests (compute_all, snapshots) expect it
        property_registry._store.pop("_pydantic_test_property", None)


def test_json_schema_validates_the_format():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json_schema()
    c = (Distance("JUICE_JANUS", "GANYMEDE") > "2 km") & Inverted(
        PhaseAngle("JUICE_JANUS", "GANYMEDE") < "20 deg"
    )
    data = json.loads(json.dumps(unstructure_node(c)))
    jsonschema.validate(data, schema)

    data["left"]["left"]["observer"] = 3
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


def test_time_segment_is_frozen_and_comparable():
    seg = TimeSegment("2032-01-01", "2032-01-02", label="a")
    with pytest.raises(FrozenInstanceError):
        seg.label = "b"  # type: ignore[misc]
    other = TimeSegment("2032-01-01", "2032-01-02", label="b")
    assert seg != other
    assert seg.equals(other, exclude="label")


def test_collection_as_pandas():
    seg = TimeSegment("2032-01-01", "2032-01-02", metadata={"flyby": "G1"})
    df = TimeSegmentsCollection(segments=[seg]).as_pandas()
    assert list(df.columns) == [
        "start",
        "end",
        "label",
        "value",
        "property_name",
        "metadata.flyby",
        "duration",
    ]
    assert df["duration"].iloc[0] == pd.Timedelta(days=1)
