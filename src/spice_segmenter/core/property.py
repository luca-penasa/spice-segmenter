from __future__ import annotations
from typing import ClassVar
from typing import TYPE_CHECKING

from ..support.time_types import TIMES_TYPES

if TYPE_CHECKING:
    from ..ops.unit_adapter import UnitAdaptor
    from .constraints import Constraint, left_types

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum, auto

import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define
from loguru import logger as log
from spiceypy.utils.callbacks import UDFUNB, UDFUNS


class PropertyTypes(Enum):
    SCALAR = auto()
    BOOLEAN = auto()
    VECTOR = auto()
    DISCRETE = auto()


@define(repr=False, order=False, eq=False)
class Property(ABC):
    _name: ClassVar[str]
    _unit: ClassVar[pint.Unit | Iterable[pint.Unit]]
    _type: ClassVar[PropertyTypes] = PropertyTypes.SCALAR

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register property subclasses automatically in the global registry."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_name") and isinstance(cls._name, str) and cls._name:
            from spice_segmenter.core.registry import property_registry
            if cls.__dict__.get("_generated_by_property_function", False):
                # Alternative / functional implementation — defer to the first
                # registration if the name is already taken.
                property_registry.register_or_skip(cls._name, cls)
            else:
                property_registry.register(cls._name, cls)

    @abstractmethod
    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum: ...

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def name(self) -> str:
        """Property name, read from _name class attribute."""
        return self._name

    @property
    def unit(self) -> pint.Unit | Iterable[pint.Unit]:
        """Property unit, read from _unit class attribute."""
        return self._unit

    @property
    def type(self) -> PropertyTypes:
        """Property type, read from _type class attribute or default to SCALAR."""
        return getattr(self.__class__, "_type", PropertyTypes.SCALAR)

    def as_unit(self, unit: pint.Unit | str) -> UnitAdaptor:
        from ..ops.unit_adapter import UnitAdaptor

        return UnitAdaptor(self, unit)

    def has_unit(self) -> bool:
        return bool(str(self.unit))

    @property
    def instance_id(self) -> str:
        """Human-readable, instance-unique identifier for this property.

        Built from the property name and the values of any attrs fields that
        carry a meaningful identity (SPICE body/instrument references and plain
        strings).  Fields that are purely computational (default corrections,
        numeric parameters) are included only when they differ from their
        default value to keep the id concise.

        Examples
        --------
        >>> Distance("JUICE_JANUS", "GANYMEDE").instance_id
        'juice_janus_ganymede_distance'
        >>> PhaseAngle("JUICE_JANUS", "TITAN").instance_id
        'juice_janus_titan_phase_angle'
        >>> Occultation("JUICE", "GANYMEDE", "JUPITER").instance_id
        'juice_ganymede_jupiter_occultation'
        """
        import re
        import attrs

        def _normalise(value: object) -> str | None:
            """Return a slug for *value*, or None to skip it."""
            # SpiceRef-like objects expose a .name attribute
            name = getattr(value, "name", None)
            if isinstance(name, str) and name:
                return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            # Plain strings (e.g. light_time_correction) — only include if
            # non-trivial (not "NONE" / empty)
            if isinstance(value, str):
                norm = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
                return norm if norm and norm != "none" else None
            return None

        parts: list[str] = []
        try:
            for f in attrs.fields(type(self)):
                if not f.init or f.name.startswith("_"):
                    continue
                val = getattr(self, f.name)
                # Skip string fields that are at their default value — they are
                # method/mode selectors that don't distinguish instances.
                if isinstance(val, str) and f.default is not attrs.NOTHING:
                    default = f.default() if callable(f.default) else f.default
                    if val == default:
                        continue
                slug = _normalise(val)
                if slug:
                    parts.append(slug)
        except attrs.exceptions.NotAnAttrsClassError:
            pass

        parts.append(re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_"))
        return "_".join(parts)

    def compute_as_spice_function(self) -> UDFUNS:
        def as_function(time: TIMES_TYPES) -> float | bool | Enum:
            return self.__call__(time)

        # TODO we are marking as_function as returing float, bool or enum, but wont work with SpiceUDFUNS. You need to use SpiceUDFUNB instead for booleans
        # while for enum wont work at all. Move these routines in the derived ScalarProperty and BooleanProperty classes please!
        return spiceypy.utils.callbacks.SpiceUDFUNS(as_function)

    def is_decreasing(self, time: TIMES_TYPES) -> bool:
        return spiceypy.uddc(self.compute_as_spice_function(), time, self.dt)  # type: ignore

    def is_decreasing_as_spice_function(self) -> UDFUNB:
        def as_function(function: Callable, time: TIMES_TYPES) -> bool:
            return self.is_decreasing(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)

    def __repr__(self) -> str:
        return f"{self.name}"

    def _handle_other_operand(self, other: left_types) -> Property:
        if isinstance(other, Property):
            return other

        from ..ops.constant_values import Constant

        return Constant.from_value(other)

    def __gt__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        return Constraint(self, self._handle_other_operand(other), ">")

    def __ge__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        log.warning(
            "Using >= operator on properties is not supported by SPICE. Using > instead."
        )
        return Constraint(self, self._handle_other_operand(other), ">")

    def __le__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        log.warning(
            "Using <= operator on properties is not supported by SPICE. Using < instead."
        )
        return Constraint(self, self._handle_other_operand(other), "<")

    def __lt__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        return Constraint(self, self._handle_other_operand(other), "<")

    def __and__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        return Constraint(self, self._handle_other_operand(other), "&")

    def __eq__(self, other: left_types) -> Constraint:  # type: ignore
        # Check if comparing with MinMaxConditionTypes enum for convenient syntax
        from ..properties.observation_properties import MinMaxConditionTypes

        if isinstance(other, MinMaxConditionTypes):
            from ..ops.constraint_operations import MinMaxConstraint

            return MinMaxConstraint(self, other)

        other = self._handle_other_operand(other)
        if not isinstance(other, Property):
            return NotImplemented

        from .constraints import Constraint

        return Constraint(self, other, "=")

    def __or__(self, other: left_types) -> Constraint:
        from .constraints import Constraint

        return Constraint(self, self._handle_other_operand(other), "|")

    def config(self, config: dict) -> None:
        log.debug("adding prop unit for {}", self.unit)
        config["property_unit"] = str(self.unit)
        config["property"] = self.name

    def to_json(self, indent: int | None = None) -> str:
        """Serialize this Property to a JSON string.

        Parameters
        ----------
        indent:
            Number of spaces for JSON indentation. ``None`` for compact format.
        """
        import json

        from ..support.serialization import create_property_converter

        converter = create_property_converter()
        data = converter.unstructure(self)
        return json.dumps(data, indent=indent)

    def find_minimum(
        self,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool = True,
    ) -> "TimeSegment":  # type: ignore[name-defined]  # noqa: F821
        """Find the global minimum of this property within *window*.

        Parameters
        ----------
        window:
            Search window as a :class:`~spice_segmenter.core.TimeSegmentsCollection`.
        evaluate:
            If ``True`` (default) the property is evaluated at the result time
            and stored in :attr:`~TimeSegment.value`.

        Returns
        -------
        TimeSegment
            A zero-duration point segment with :attr:`~TimeSegment.is_point` ``= True``.

        Examples
        --------
        >>> ca = Distance("JUICE_JANUS", "GANYMEDE").find_minimum(window)
        >>> ca.time, ca.value   # timestamp and distance in km
        """
        return self._find_extremum("GLOBAL_MINIMUM", window, evaluate=evaluate)

    def find_maximum(
        self,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool = True,
    ) -> "TimeSegment":
        """Find the global maximum of this property within *window*."""
        return self._find_extremum("GLOBAL_MAXIMUM", window, evaluate=evaluate)

    def find_local_minima(
        self,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool = True,
    ) -> "TimeSegmentsCollection":
        """Find all local minima of this property within *window*.

        Returns a :class:`~spice_segmenter.core.TimeSegmentsCollection` of
        zero-duration point segments, one per local minimum.

        Examples
        --------
        >>> minima = Distance("JUICE_JANUS", "GANYMEDE").find_local_minima(window)
        >>> minima.point_events    # list of point segments
        """
        return self._find_extrema("LOCAL_MINIMUM", window, evaluate=evaluate)

    def find_local_maxima(
        self,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool = True,
    ) -> "TimeSegmentsCollection":
        """Find all local maxima of this property within *window*."""
        return self._find_extrema("LOCAL_MAXIMUM", window, evaluate=evaluate)

    def _find_extremum(
        self,
        condition: str,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool,
    ) -> "TimeSegment":
        """Solve a global min/max condition and return a single point segment."""
        results = self._find_extrema(condition, window, evaluate=evaluate)
        pts = results.point_events
        if not pts:
            raise ValueError(
                f"Could not find {condition} for {self!r} in the given window."
            )
        if len(pts) > 1:
            log.warning(
                "find_minimum/maximum returned {} point events; returning the first.",
                len(pts),
            )
        return pts[0]

    def _find_extrema(
        self,
        condition: str,
        window: "TimeSegmentsCollection" = None,
        *,
        evaluate: bool,
    ) -> "TimeSegmentsCollection":
        """Solve a MinMaxConstraint and return a TimeSegmentsCollection of point segments."""
        from ..ops.constraint_operations import MinMaxConstraint
        from ..properties.observation_properties import MinMaxConditionTypes
        from ..support.config import get_active_config
        from .time_segment import TimeSegment
        from .time_segments_collection import TimeSegmentsCollection

        if window is None:
            cfg = get_active_config()
            if cfg.start is None or cfg.end is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{condition.lower()} called without a window "
                    "and no default window is set. "
                    "Either pass a window explicitly or activate a context:\n\n"
                    "    with Config(start='2032-01-01', end='2035-01-01'):\n"
                    f"        prop.find_{condition.lower()}()\n"
                )
            window = TimeSegmentsCollection.from_start_end(cfg.start, cfg.end)

        cond = MinMaxConditionTypes[condition]
        constraint = MinMaxConstraint(self, cond)
        raw: TimeSegmentsCollection = constraint.solve(window)

        # Annotate each point segment with property metadata + evaluated value
        annotated = []
        for seg in raw:
            val = float(self(seg.start)) if evaluate else None
            annotated.append(
                TimeSegment.at_time(
                    seg.start,
                    label=condition.lower().replace("_", " "),
                    value=val,
                    property_name=self.name,
                )
            )
        return TimeSegmentsCollection(segments=annotated)

    @classmethod
    def from_json(cls, json_str: str) -> Property:
        """
        Deserialize a Property from a JSON string.

        Args:
            json_str: JSON string representation of a Property

        Returns:
            Reconstructed Property object

        Example:
            >>> json_str = '{"type": "PhaseAngle", ...}'
            >>> prop = Property.from_json(
            ...     json_str
            ... )
        """
        import json

        from ..support.serialization import create_property_converter

        converter = create_property_converter()
        data = json.loads(json_str)
        return converter.structure(data, cls)


@define(repr=False, order=False, eq=False)
class BooleanProperty(Property):
    @abstractmethod
    def __call__(self, time: TIMES_TYPES) -> bool:
        pass

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

    @property
    def unit(self) -> pint.Unit:
        return pint.Unit("")

    def has_unit(self) -> bool:
        return False

    def compute_as_spice_function(self, invert: bool = False) -> UDFUNB:
        if invert:

            def as_function(udfun, time: TIMES_TYPES) -> bool:
                return ~self.__call__(time)

        else:

            def as_function(udfun, time: TIMES_TYPES) -> bool:
                return self.__call__(time)

        return spiceypy.utils.callbacks.SpiceUDFUNB(as_function)
