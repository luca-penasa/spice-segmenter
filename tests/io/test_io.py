"""Tests for the spice_segmenter.io subpackage.

Covers:
- DSL parse + constraint_to_expression round-trips
- YAML constraint: loads / dumps / load (from file)
- YAML properties: loads_properties / dumps_properties / load_properties
- Variety of property types: simple, optional-field overrides, required-extra fields,
  per-entry context overrides, multi-target lists
- Backwards-compat shims in support.*
- Error paths: unknown type, missing type key, bad YAML shape
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spice_segmenter.io import (
    constraint_to_context,
    constraint_to_expression,
    dump,
    dump_properties,
    dumps,
    dumps_properties,
    load,
    load_properties,
    loads,
    loads_properties,
    parse,
)
from spice_segmenter.properties.observation_properties import (
    AngularSize,
    Distance,
    PhaseAngle,
)
from spice_segmenter.properties.occultation_types import Occultation, OccultationTypes
from spice_segmenter.properties.reflector_properties import ShineProperties

from tests import tour_config  # noqa: F401 — loads SPICE kernels as a side-effect

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS = "JUICE_JANUS"
TGT = "GANYMEDE"
CTX = {"observer": OBS, "target": TGT, "light_time_correction": "NONE"}


# ---------------------------------------------------------------------------
# DSL — parse
# ---------------------------------------------------------------------------

class TestDslParse:
    """DSL expression parsing."""

    def test_scalar_comparison_gt(self):
        c = parse("distance > '1e4 km'", context=CTX)
        assert c is not None
        assert constraint_to_expression(c) == "distance > '10000.0 km'"

    def test_scalar_comparison_lt(self):
        c = parse("phase_angle < '30 deg'", context=CTX)
        expr = constraint_to_expression(c)
        assert "phase_angle" in expr
        assert "<" in expr

    def test_boolean_shorthand(self):
        """Bare property name → prop == True shorthand."""
        c = parse("fov_visibility", context=CTX)
        expr = constraint_to_expression(c)
        assert expr == "fov_visibility"

    def test_invert_tilde(self):
        c = parse("~fov_visibility", context=CTX)
        expr = constraint_to_expression(c)
        assert "~" in expr and "fov_visibility" in expr

    def test_invert_not(self):
        c1 = parse("~fov_visibility", context=CTX)
        c2 = parse("not fov_visibility", context=CTX)
        assert constraint_to_expression(c1) == constraint_to_expression(c2)

    def test_boolean_or(self):
        """or operator — natural Python precedence, no extra parens needed."""
        c = parse("distance > '1e4 km' or phase_angle < '30 deg'", context=CTX)
        expr = constraint_to_expression(c)
        assert "|" in expr

    def test_boolean_and(self):
        c = parse(
            "(distance > '1e4 km' or phase_angle < '30 deg') and fov_visibility",
            context=CTX,
        )
        expr = constraint_to_expression(c)
        assert "&" in expr and "|" in expr and "fov_visibility" in expr

    def test_bitwise_or_and(self):
        """Bitwise | and & require parens around comparisons — same result."""
        c1 = parse(
            "(distance > '1e4 km' or phase_angle < '30 deg') and fov_visibility",
            context=CTX,
        )
        c2 = parse(
            "((distance > '1e4 km') | (phase_angle < '30 deg')) & fov_visibility",
            context=CTX,
        )
        assert constraint_to_expression(c1) == constraint_to_expression(c2)

    def test_subscript_component(self):
        c = parse(
            "shine_properties[0] > '5 deg'",
            context=CTX,
            overrides={"shine_properties": {"reflector": "JUPITER"}},
        )
        expr = constraint_to_expression(c)
        assert "shine_properties[0]" in expr

    def test_named_sub_property(self):
        c = parse(
            "shine_properties.reflector_elevation > '5 deg'",
            context=CTX,
            overrides={"shine_properties": {"reflector": "JUPITER"}},
        )
        expr = constraint_to_expression(c)
        assert "shine_properties.reflector_elevation" in expr

    def test_optional_field_override(self):
        """phase_angle with a non-default third_body."""
        c = parse(
            "phase_angle < '30 deg'",
            context=CTX,
            overrides={"phase_angle": {"third_body": "JUPITER"}},
        )
        expr = constraint_to_expression(c)
        assert "phase_angle" in expr

    def test_unknown_property_raises(self):
        with pytest.raises(KeyError, match="Unknown property"):
            parse("nonexistent_prop > '1 km'", context=CTX)

    def test_invalid_syntax_raises(self):
        with pytest.raises(SyntaxError):
            parse("distance > > '1 km'", context=CTX)


# ---------------------------------------------------------------------------
# DSL — constraint_to_expression round-trips
# ---------------------------------------------------------------------------

class TestConstraintToExpression:
    """Serialization of Constraint objects to DSL strings."""

    def test_simple_distance(self):
        c = Distance(**CTX) > "1e4 km"
        expr = constraint_to_expression(c)
        assert expr == "distance > '10000.0 km'"

    def test_simple_phase_angle(self):
        c = PhaseAngle(**CTX) < "30 deg"
        expr = constraint_to_expression(c)
        assert "phase_angle" in expr and "<" in expr

    def test_boolean_true_shorthand(self):
        from spice_segmenter.properties.visibility_properties import BodyFOVVisibility
        c = BodyFOVVisibility(**CTX) == True  # noqa: E712
        expr = constraint_to_expression(c)
        assert "fov_visibility" == expr

    def test_compound_round_trip(self):
        c = (Distance(**CTX) > "1e4 km") | (PhaseAngle(**CTX) < "30 deg")
        expr = constraint_to_expression(c)
        c2 = parse(expr, context=CTX)
        assert constraint_to_expression(c2) == expr

    def test_inverted_round_trip(self):
        from spice_segmenter.properties.visibility_properties import BodyFOVVisibility
        c = ~(BodyFOVVisibility(**CTX) == True)  # noqa: E712
        expr = constraint_to_expression(c)
        c2 = parse(expr, context=CTX)
        assert constraint_to_expression(c2) == expr

    def test_context_extraction_simple(self):
        c = Distance(**CTX) > "1e4 km"
        ctx, overrides = constraint_to_context(c)
        assert ctx["observer"] == OBS
        assert ctx["target"] == TGT
        assert overrides == {}

    def test_context_extraction_with_overrides(self):
        p = PhaseAngle(OBS, TGT, third_body="JUPITER", light_time_correction="NONE")
        c = p < "30 deg"
        ctx, overrides = constraint_to_context(c)
        assert ctx["observer"] == OBS
        assert "phase_angle" in overrides
        assert overrides["phase_angle"].get("third_body") is not None


# ---------------------------------------------------------------------------
# YAML — constraint round-trips
# ---------------------------------------------------------------------------

class TestConstraintYaml:
    """YAML serialization/deserialisation of constraints."""

    def _simple(self):
        return Distance(**CTX) > "1e4 km"

    def _compound(self):
        return (Distance(**CTX) > "1e4 km") | (PhaseAngle(**CTX) < "30 deg")

    def test_dumps_is_string(self):
        assert isinstance(dumps(self._simple()), str)

    def test_dumps_contains_expression_key(self):
        s = dumps(self._simple())
        assert "expression:" in s

    def test_dumps_contains_context(self):
        s = dumps(self._simple())
        assert OBS in s and TGT in s

    def test_loads_round_trip_simple(self):
        c = self._simple()
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_loads_round_trip_compound(self):
        c = self._compound()
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_loads_round_trip_inverted(self):
        from spice_segmenter.properties.visibility_properties import BodyFOVVisibility
        c = ~(BodyFOVVisibility(**CTX) == True)  # noqa: E712
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_loads_round_trip_with_overrides(self):
        p = PhaseAngle(OBS, TGT, third_body="JUPITER", light_time_correction="NONE")
        c = p < "30 deg"
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_load_from_file(self, tmp_path):
        c = self._compound()
        path = tmp_path / "constraint.yaml"
        dump(c, path)
        c2 = load(path)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_load_fixture_compound_constraint(self):
        """Load the bundled compound_constraint.yaml fixture."""
        c = load(FIXTURES / "compound_constraint.yaml")
        expr = constraint_to_expression(c)
        assert "distance" in expr
        assert "phase_angle" in expr
        assert "fov_visibility" in expr

    def test_loads_missing_expression_raises(self):
        with pytest.raises(ValueError, match="expression"):
            loads("context:\n  observer: JUICE_JANUS\n")

    def test_loads_bad_shape_raises(self):
        with pytest.raises(ValueError):
            loads("- foo\n- bar\n")


# ---------------------------------------------------------------------------
# YAML — property-list round-trips
# ---------------------------------------------------------------------------

class TestPropertyListYaml:
    """YAML serialization/deserialisation of property lists."""

    def _simple_list(self):
        return [Distance(**CTX), PhaseAngle(**CTX), AngularSize(**CTX)]

    def test_dumps_is_string(self):
        assert isinstance(dumps_properties(self._simple_list()), str)

    def test_dumps_contains_type_keys(self):
        s = dumps_properties(self._simple_list())
        assert "type: distance" in s
        assert "type: phase_angle" in s

    def test_dumps_shared_context_promoted(self):
        """When all properties share observer/target they go to context block."""
        s = dumps_properties(self._simple_list())
        lines = s.splitlines()
        # context: block should appear before any "- type:" line
        ctx_idx = next(i for i, l in enumerate(lines) if "context:" in l)
        type_idx = next(i for i, l in enumerate(lines) if "- type:" in l)
        assert ctx_idx < type_idx
        assert OBS in s and TGT in s

    def test_round_trip_simple(self):
        props = self._simple_list()
        yaml_str = dumps_properties(props)
        props2 = loads_properties(yaml_str)
        assert len(props) == len(props2)
        for p, p2 in zip(props, props2):
            assert type(p) is type(p2)
            assert str(p.observer) == str(p2.observer)
            assert str(p.target) == str(p2.target)

    def test_round_trip_optional_field(self):
        """phase_angle with non-default third_body survives round-trip."""
        p = PhaseAngle(OBS, TGT, third_body="JUPITER", light_time_correction="NONE")
        yaml_str = dumps_properties([p])
        props2 = loads_properties(yaml_str)
        assert len(props2) == 1
        assert type(props2[0]) is PhaseAngle
        assert str(props2[0].third_body) == "JUPITER"

    def test_round_trip_required_extra_field(self):
        """shine_properties requires 'reflector' — must survive round-trip."""
        p = ShineProperties(OBS, TGT, reflector="JUPITER", light_time_correction="NONE")
        yaml_str = dumps_properties([p])
        props2 = loads_properties(yaml_str)
        assert len(props2) == 1
        assert type(props2[0]) is ShineProperties
        assert str(props2[0].reflector) == "JUPITER"

    def test_round_trip_required_and_optional_extra_fields(self):
        p = ShineProperties(
            OBS, TGT, reflector="JUPITER", light_source="CALLISTO",
            light_time_correction="NONE"
        )
        yaml_str = dumps_properties([p])
        props2 = loads_properties(yaml_str)
        p2 = props2[0]
        assert str(p2.reflector) == "JUPITER"
        assert str(p2.light_source) == "CALLISTO"

    def test_round_trip_different_targets(self):
        """Properties with different targets — target NOT promoted to shared context."""
        p1 = Distance(OBS, "GANYMEDE", light_time_correction="NONE")
        p2 = Distance(OBS, "CALLISTO", light_time_correction="NONE")
        yaml_str = dumps_properties([p1, p2])
        # target must not appear in shared context block
        data_lines = yaml_str.splitlines()
        # find where 'context:' block ends (next top-level key)
        in_ctx = False
        ctx_lines = []
        for line in data_lines:
            if line.startswith("context:"):
                in_ctx = True
                continue
            if in_ctx:
                if line.startswith(" ") or line.startswith("\t"):
                    ctx_lines.append(line)
                else:
                    break
        assert all("target" not in l for l in ctx_lines)
        # round-trip still correct
        props2 = loads_properties(yaml_str)
        assert str(props2[0].target) == "GANYMEDE"
        assert str(props2[1].target) == "CALLISTO"

    def test_load_from_file(self, tmp_path):
        props = self._simple_list()
        path = tmp_path / "props.yaml"
        dump_properties(props, path)
        props2 = load_properties(path)
        assert [type(p) for p in props] == [type(p) for p in props2]

    def test_load_fixture_simple_properties(self):
        """Load the bundled simple_properties.yaml fixture."""
        props = load_properties(FIXTURES / "simple_properties.yaml")
        assert len(props) == 5
        types = {type(p).__name__ for p in props}
        assert "Distance" in types
        assert "PhaseAngle" in types
        assert "AngularSize" in types

    def test_load_fixture_mixed_properties(self):
        """Load the bundled mixed_properties.yaml fixture."""
        props = load_properties(FIXTURES / "mixed_properties.yaml")
        assert len(props) == 7
        # First prop: distance/GANYMEDE
        assert type(props[0]).__name__ == "Distance"
        assert str(props[0].target) == "GANYMEDE"
        # Third entry: distance/CALLISTO (per-entry target override)
        assert type(props[2]).__name__ == "Distance"
        assert str(props[2].target) == "CALLISTO"
        # Phase angle with JUPITER third_body override
        assert type(props[1]).__name__ == "PhaseAngle"
        assert str(props[1].third_body) == "JUPITER"
        # shine_properties with required reflector
        shine = props[4]
        assert type(shine).__name__ == "ShineProperties"
        assert str(shine.reflector) == "JUPITER"
        # shine_properties with non-default light_source
        shine2 = props[5]
        assert str(shine2.light_source) == "CALLISTO"
        # angular_separation with required other
        assert type(props[6]).__name__ == "AngularSeparation"
        assert str(props[6].other) == "CALLISTO"

    def test_loads_missing_properties_raises(self):
        with pytest.raises(ValueError, match="properties"):
            loads_properties("context:\n  observer: JUICE_JANUS\n")

    def test_loads_properties_not_list_raises(self):
        with pytest.raises(ValueError, match="list"):
            loads_properties(
                "properties:\n  distance:\n    target: GANYMEDE\n"
            )

    def test_loads_entry_not_dict_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            loads_properties("properties:\n  - distance\n")

    def test_loads_missing_type_key_raises(self):
        with pytest.raises(ValueError, match="type"):
            loads_properties(
                "properties:\n  - target: GANYMEDE\n"
            )

    def test_loads_unknown_property_type_raises(self):
        with pytest.raises(KeyError, match="Unknown property"):
            loads_properties("properties:\n  - type: nonexistent_prop\n")


# ---------------------------------------------------------------------------
# Occultation — property-list and constraint YAML
# ---------------------------------------------------------------------------

class TestOccultation:
    """Occultation is special: no target field, observer has no factory default,
    front and back are required non-context fields."""

    OBS = "JUICE"

    def _prop(self, front="GANYMEDE", back="JUPITER", ltc="NONE"):
        return Occultation(self.OBS, front, back, light_time_correction=ltc)

    # ── Registry ─────────────────────────────────────────────────────────

    def test_occultation_registered(self):
        from spice_segmenter.core.registry import all as all_props
        assert "occultation" in all_props()

    # ── Property-list round-trips ─────────────────────────────────────────

    def test_dumps_single(self):
        yaml_str = dumps_properties([self._prop()])
        assert "type: occultation" in yaml_str
        assert "front: GANYMEDE" in yaml_str
        assert "back: JUPITER" in yaml_str

    def test_round_trip_single(self):
        p = self._prop()
        props2 = loads_properties(dumps_properties([p]))
        p2 = props2[0]
        assert type(p2) is Occultation
        assert str(p2.observer) == self.OBS
        assert str(p2.front) == "GANYMEDE"
        assert str(p2.back) == "JUPITER"
        assert p2.light_time_correction == "NONE"

    def test_round_trip_multiple_different_front(self):
        """Multiple occultations with same back but different fronts."""
        props = [self._prop("GANYMEDE"), self._prop("CALLISTO")]
        props2 = loads_properties(dumps_properties(props))
        assert str(props2[0].front) == "GANYMEDE"
        assert str(props2[1].front) == "CALLISTO"

    def test_round_trip_non_default_ltc(self):
        p = self._prop(ltc="LT")
        props2 = loads_properties(dumps_properties([p]))
        assert props2[0].light_time_correction == "LT"

    def test_shared_observer_promoted(self):
        """observer should be in shared context when all entries agree."""
        props = [self._prop("GANYMEDE"), self._prop("CALLISTO")]
        yaml_str = dumps_properties(props)
        # 'observer: JUICE' must appear under 'context:', not under each entry
        lines = yaml_str.splitlines()
        ctx_section = []
        in_ctx = False
        for line in lines:
            if line.startswith("context:"):
                in_ctx = True
                continue
            if in_ctx:
                if line.startswith(" ") or line.startswith("\t"):
                    ctx_section.append(line)
                else:
                    break
        assert any(self.OBS in l for l in ctx_section), \
            f"observer not promoted to shared context: {yaml_str}"

    def test_load_fixture_occultation_properties(self):
        props = load_properties(FIXTURES / "occultation_properties.yaml")
        assert len(props) == 3
        assert all(type(p) is Occultation for p in props)
        assert str(props[0].front) == "GANYMEDE"
        assert str(props[1].front) == "CALLISTO"
        assert props[2].light_time_correction == "LT"

    # ── Constraint round-trips ────────────────────────────────────────────

    def test_constraint_expression(self):
        c = self._prop() == OccultationTypes.FULL
        expr = constraint_to_expression(c)
        assert expr == "occultation == OccultationTypes.FULL"

    def test_constraint_context_extraction(self):
        c = self._prop() == OccultationTypes.FULL
        ctx, overrides = constraint_to_context(c)
        assert ctx["observer"] == self.OBS
        assert "target" not in ctx            # Occultation has no target field
        assert overrides["occultation"]["front"] == "GANYMEDE"
        assert overrides["occultation"]["back"] == "JUPITER"

    def test_constraint_yaml_round_trip(self):
        c = self._prop() == OccultationTypes.FULL
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_constraint_yaml_contains_keys(self):
        c = self._prop() == OccultationTypes.FULL
        yaml_str = dumps(c)
        assert "occultation" in yaml_str
        assert "GANYMEDE" in yaml_str
        assert "JUPITER" in yaml_str
        assert "OccultationTypes.FULL" in yaml_str

    def test_constraint_any_type(self):
        c = self._prop() == OccultationTypes.ANY
        yaml_str = dumps(c)
        c2 = loads(yaml_str)
        assert "OccultationTypes.ANY" in constraint_to_expression(c2)

    def test_load_fixture_occultation_constraint(self):
        c = load(FIXTURES / "occultation_constraint.yaml")
        expr = constraint_to_expression(c)
        assert "occultation" in expr
        assert "OccultationTypes.FULL" in expr

    def test_parse_occultation_expression(self):
        c = parse(
            "occultation == OccultationTypes.FULL",
            context={"observer": self.OBS, "light_time_correction": "NONE"},
            overrides={"occultation": {"front": "GANYMEDE", "back": "JUPITER"}},
        )
        assert constraint_to_expression(c) == "occultation == OccultationTypes.FULL"

    def test_compound_occultation_constraint(self):
        """Occultation combined with another constraint."""
        c = (self._prop("GANYMEDE") == OccultationTypes.FULL) | (
            self._prop("CALLISTO") == OccultationTypes.ANY
        )
        yaml_str = dumps(c)
        # Conflicting observers? No — same observer. Should round-trip.
        c2 = loads(yaml_str)
        assert constraint_to_expression(c) == constraint_to_expression(c2)


# ---------------------------------------------------------------------------
# Top-level package re-exports
# ---------------------------------------------------------------------------

class TestTopLevelReExports:
    """Functions exported from spice_segmenter root still work."""

    def test_parse_constraint(self):
        from spice_segmenter import parse_constraint
        c = parse_constraint("distance > '1e4 km'", context=CTX)
        assert c is not None

    def test_dumps_loads_constraint(self):
        from spice_segmenter import dumps_constraint, loads_constraint, constraint_to_expression
        c = Distance(**CTX) > "1e4 km"
        c2 = loads_constraint(dumps_constraint(c))
        assert constraint_to_expression(c) == constraint_to_expression(c2)

    def test_dumps_loads_properties(self):
        from spice_segmenter import dumps_properties, loads_properties
        props = [Distance(**CTX), PhaseAngle(**CTX)]
        props2 = loads_properties(dumps_properties(props))
        assert [type(p) for p in props] == [type(p) for p in props2]
