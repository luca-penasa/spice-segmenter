from __future__ import annotations
from typing import ClassVar
from typing import TYPE_CHECKING

from ..support.time_types import TIMES_TYPES

if TYPE_CHECKING:
    from .constraints import Constraint, left_types

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum, auto

import numpy as np
import pint
import spiceypy
import spiceypy.utils.callbacks
from attrs import define, field
from loguru import logger as log
from spiceypy.utils.callbacks import UDFUNB, UDFUNS


def _to_pint_unit(x: pint.Unit | str | tuple | list | None) -> pint.Unit | tuple:
    """Converter for ``unit`` attrs fields: strings become ``pint.Unit``,
    tuples/lists are converted element-wise, existing ``pint.Unit`` pass through.
    """
    if isinstance(x, (list, tuple)):
        return tuple(_to_pint_unit(u) for u in x)
    if isinstance(x, str):
        return pint.Unit(x)
    if isinstance(x, pint.Unit):
        return x
    if x is None:
        raise TypeError("unit cannot be None; provide a unit string or pint.Unit")
    return pint.Unit(str(x))  # fallback for pint-compatible objects


def _bulk_et(time) -> np.ndarray:
    """Convert an array-like of times to a float64 numpy array of SPICE ETs.

    Fast paths (in order of preference):
    - ``pd.DatetimeIndex``: format to ISO strings then call ``cyice.str2et_v``
      in one C-level batch — avoids N Python→C round-trips of the scalar loop.
    - numpy float/int array: already ETs, just cast and return.
    - list/array of strings: convert to numpy string array then ``cyice.str2et_v``.
    - anything else: fall back to the original scalar ``et()`` loop.
    """
    import pandas as pd
    from spiceypy.cyice import str2et_v

    if isinstance(time, pd.DatetimeIndex):
        iso = np.array([str(t) for t in time], dtype=np.str_)
        return str2et_v(iso)

    arr = np.asarray(time)
    if arr.dtype.kind in ("f", "i", "u"):  # already numeric ET values
        return arr.astype(np.float64)
    if arr.dtype.kind in ("U", "S", "O"):  # string-like
        if arr.dtype.kind != "U":
            arr = arr.astype(str)
        return str2et_v(arr)

    # Generic fallback: scalar et() loop (handles Timestamps, mixed types, etc.)
    from ..support.spice_utilities import et as _et
    return np.array([float(_et(t)) for t in time], dtype=np.float64)


class PropertyTypes(Enum):
    SCALAR = auto()
    BOOLEAN = auto()
    VECTOR = auto()
    DISCRETE = auto()


@define(repr=False, order=False, eq=False)
class Property(ABC):
    _name: ClassVar[str]
    _unit: ClassVar[pint.Unit | tuple | None] = None  # For backward compat; instance 'unit' field takes precedence
    _type: ClassVar[PropertyTypes] = PropertyTypes.SCALAR
    # Set to a numpy.vectorize signature string (e.g. "()->(n)") on subclasses
    # whose _call_scalar returns an array rather than a scalar. Used by the
    # default _call_vector implementation.
    _vector_output_shape: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register property subclasses automatically in the global registry."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_name") and isinstance(cls._name, str) and cls._name:
            from spice_segmenter.core.registry import property_registry
            property_registry.register(cls._name, cls)

    def _call_scalar(self, time_et: float) -> float | bool | Enum:
        """Evaluate the property at a single SPICE ET (float seconds past J2000).

        Override this instead of ``__call__`` to implement a property. The base
        class ``__call__`` dispatches to this for scalar inputs and to
        ``_call_vector`` for array inputs.

        Pair with a ``_call_vector`` override that calls the corresponding
        ``cyice._v`` function to get native C-level batch performance.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented _call_scalar. "
            "Override _call_scalar with the property body (accepts a pre-converted "
            "float ET), and optionally override _call_vector with a cyice _v call "
            "for batch performance."
        )

    def _call_vector(self, times_et: np.ndarray) -> np.ndarray:
        """Evaluate at an array of SPICE ETs (float64 seconds past J2000 TDB).

        Default: ``np.vectorize`` over ``_call_scalar`` — identical behaviour to
        the old ``@vectorize`` decorator, just without the Python overhead of
        re-wrapping each call.

        Override this method with the matching ``cyice._v`` function for
        native C-level vectorisation. The input is already an ``np.ndarray`` of
        float64 ET values — no time conversion is needed inside this method.
        """
        return np.vectorize(self._call_scalar, signature=self._vector_output_shape)(times_et)

    def __call__(self, time: TIMES_TYPES) -> float | bool | Enum | np.ndarray:
        """Evaluate the property at one or more times.

        Delegates to the global :class:`~spice_segmenter.engines.Evaluator`
        so that computation is fully decoupled from the property data class.

        Subclasses that override ``__call__`` directly (e.g. with
        ``@vectorize``) continue to work: Python's MRO finds the subclass
        method first and this bridge is never reached for those classes.
        """
        from ..engines.evaluator import get_evaluator
        return get_evaluator().evaluate(self, time)

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def name(self) -> str:
        """Property name, read from _name class attribute."""
        return self._name

    @property  
    def unit(self) -> pint.Unit | tuple:
        """Property unit: check for instance field first, then fallback to class _unit."""
        import attrs
        
        # Check if 'unit' is defined as an attrs field
        try:
            fields = attrs.fields(type(self))
            for attr_field in fields:
                if attr_field.name == 'unit':
                    # 'unit' is an attrs field; get it from instance variables
                    instance_vars = vars(self)
                    if 'unit' in instance_vars:
                        return instance_vars['unit']
                    # Not in vars, use the default from field definition
                    if attr_field.default is not attrs.NOTHING:
                        return attr_field.default
                    return pint.Unit('')
        except (TypeError, AttributeError):
            # Not an attrs class at all
            pass
        
        # No 'unit' field defined, fallback to class-level _unit ClassVar
        return getattr(type(self), '_unit', pint.Unit(''))

    @property
    def type(self) -> PropertyTypes:
        """Property type, read from _type class attribute or default to SCALAR."""
        return getattr(self.__class__, "_type", PropertyTypes.SCALAR)

    def as_unit(self, unit: pint.Unit | str) -> Property:
        """Return a copy of this property that evaluates in *unit*.

        Raises ``ValueError`` if *unit* is dimensionally incompatible with the
        property's current unit.
        """
        import attrs

        target = _to_pint_unit(unit)
        current = self.unit  # Use the @property which handles fallback
        if current is not None and not isinstance(current, (list, tuple)):
            if not current.is_compatible_with(target):
                raise ValueError(
                    f"{self!r}: unit {target} is not compatible with current unit {current}"
                )
        return attrs.evolve(self, unit=target)

    def has_unit(self) -> bool:
        u = self.unit
        if u is None:
            return False
        return bool(str(u))

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
        # Uses evaluate_scalar_raw so SPICE GF callbacks always receive values
        # in the compute_unit (native unit of the registered function), not the
        # user-facing unit.  Solvers convert the refval to compute_unit themselves.
        from ..engines.evaluator import get_evaluator
        ev = get_evaluator()
        return spiceypy.utils.callbacks.SpiceUDFUNS(
            lambda t: float(ev.evaluate_scalar_raw(self, float(t)))
        )

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
        # Kept for backward compatibility with solver.
        # property_unit must be the COMPUTE unit (what SPICE natively returns),
        # so that the solver can convert the user's reference value into it.
        config["property"] = self.name

        # Try engine registry first for the compute unit
        try:
            from ..engines.evaluator import get_evaluator
            compute_unit = get_evaluator()._engine.get_compute_unit(type(self))
            if compute_unit is not None:
                config["property_unit"] = str(compute_unit)
                return
        except Exception:
            pass

        # Fallback: use the display unit
        unit = self.unit
        if unit is not None:
            config["property_unit"] = str(unit) if str(unit) else "dimensionless"

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
    # __call__ is NOT abstract — inherited Property.__call__ delegation bridge
    # dispatches through the evaluator for registered properties, and Python
    # MRO ensures subclass overrides are still found first for legacy classes.
    unit: pint.Unit = field(
        default=pint.Unit(""), kw_only=True, converter=_to_pint_unit
    )

    @property
    def type(self) -> PropertyTypes:
        return PropertyTypes.BOOLEAN

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
