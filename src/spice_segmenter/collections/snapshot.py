"""Auto-compute all registered properties for a given observer / target / time.

Usage
-----
::

    from spice_segmenter.collections.snapshot import compute_all

    snap = compute_all("JUICE_JANUS", "GANYMEDE", time="2032-01-01T12:00:00")
    print(snap)                   # pretty table
    df = snap.to_series()         # pandas Series (scalar properties only)
    print(snap["distance"])       # value by property name
    print(snap.units["distance"]) # unit string

"""

from __future__ import annotations

from typing import Any

import pint


class PropertySnapshot:
    """Computed values for all auto-instantiable properties at a single time.

    Attributes
    ----------
    observer : str
        Observer name used for all computations.
    target : str
        Target body name used for all computations.
    time : object
        The evaluation time (as passed by the caller).
    values : dict[str, Any]
        Mapping of ``instance_id → computed value``.
    units : dict[str, str]
        Mapping of ``instance_id → unit string``.
    properties : dict[str, Property]
        Mapping of ``instance_id → Property instance``.
    errors : dict[str, str]
        Mapping of ``instance_id → error message`` for properties that
        raised during evaluation (e.g. kernels not loaded for that body).
    """

    def __init__(
        self,
        observer: str,
        target: str,
        time: Any,
        values: dict[str, Any],
        units: dict[str, str],
        properties: dict[str, Any],
        errors: dict[str, str],
    ) -> None:
        self.observer = observer
        self.target = target
        self.time = time
        self.values = values
        self.units = units
        self.properties = properties
        self.errors = errors

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Access a computed value by property name or instance_id.

        Tries ``key`` as an ``instance_id`` first, then searches for
        a matching ``_name`` among the stored properties.
        """
        if key in self.values:
            return self.values[key]
        # fall back: search by property _name
        for iid, prop in self.properties.items():
            if prop.name == key:
                return self.values[iid]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if key in self.values:
            return True
        return any(p.name == key for p in self.properties.values())

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_series(self, scalar_only: bool = True):
        """Return a :class:`pandas.Series` of computed values.

        Parameters
        ----------
        scalar_only:
            When ``True`` (default) exclude vector / array-valued properties
            so the Series has a uniform dtype.  Set to ``False`` to include
            all ``object``-typed entries.
        """
        import numpy as np
        import pandas as pd

        data: dict[str, Any] = {}
        for iid, val in self.values.items():
            # Unwrap 0-d numpy object arrays (e.g. vectorized Enum results)
            if isinstance(val, np.ndarray) and val.ndim == 0 and val.dtype == object:
                val = val.item()
            if scalar_only and isinstance(val, np.ndarray) and val.ndim > 0:
                continue
            data[iid] = val
        return pd.Series(data, name=str(self.time))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of ``{instance_id: value}``."""
        return dict(self.values)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        import numpy as np

        lines = [
            f"PropertySnapshot  observer={self.observer!r}  "
            f"target={self.target!r}  time={self.time!r}",
            f"  {len(self.values)} computed, {len(self.errors)} errors",
            "",
        ]
        col_id = max((len(k) for k in self.values), default=10)
        col_unit = max((len(u) for u in self.units.values()), default=4)
        col_id = max(col_id, 11)
        col_unit = max(col_unit, 4)
        lines.append(
            f"  {'instance_id':<{col_id}}  {'unit':<{col_unit}}  value",
        )
        lines.append(f"  {'-'*col_id}  {'-'*col_unit}  -----")
        for iid, val in self.values.items():
            unit = self.units.get(iid, "")
            # Unwrap 0-d numpy object arrays (e.g. vectorized Enum results)
            if isinstance(val, np.ndarray) and val.ndim == 0 and val.dtype == object:
                val = val.item()
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                val_str = f"[{', '.join(f'{v:.4g}' for v in val[:4])}{'…' if len(val) > 4 else ''}]"
            elif isinstance(val, (float, np.floating, np.ndarray)):
                val_str = f"{float(val):.6g}"
            elif hasattr(val, "name"):  # Enum (e.g. OccultationTypes)
                val_str = str(val.name)
            else:
                val_str = str(val)
            lines.append(f"  {iid:<{col_id}}  {unit:<{col_unit}}  {val_str}")

        if self.errors:
            lines.append("")
            lines.append(f"  Errors ({len(self.errors)}):")
            for iid, msg in self.errors.items():
                lines.append(f"    {iid}: {msg}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML table for display in Jupyter notebooks."""
        import numpy as np

        rows = ""
        for iid, val in self.values.items():
            unit_str = self.units.get(iid, "")
            # Unwrap 0-d numpy object arrays (e.g. vectorized Enum results)
            if isinstance(val, np.ndarray) and val.ndim == 0 and val.dtype == object:
                val = val.item()
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                val_str = f"[{', '.join(f'{v:.4g}' for v in val[:4])}{'…' if len(val) > 4 else ''}]"
            elif isinstance(val, (float, np.floating, np.ndarray)):
                val_str = f"{float(val):.6g}"
            elif hasattr(val, "name"):  # Enum (e.g. OccultationTypes)
                val_str = str(val.name)
            else:
                val_str = str(val)
            rows += (
                f"<tr>"
                f"<td><strong>{iid}</strong></td>"
                f"<td><code>{unit_str}</code></td>"
                f"<td>{val_str}</td>"
                f"</tr>"
            )

        err_rows = ""
        for iid, msg in self.errors.items():
            err_rows += (
                f"<tr style='color:tomato'>"
                f"<td><strong>{iid}</strong></td>"
                f"<td colspan='2'>{msg}</td>"
                f"</tr>"
            )

        return (
            f"<b>PropertySnapshot</b> "
            f"observer=<code>{self.observer}</code> "
            f"target=<code>{self.target}</code> "
            f"time=<code>{self.time}</code><br>"
            f"<table border='1' style='border-collapse:collapse'>"
            f"<thead><tr><th>instance_id</th><th>unit</th><th>value</th></tr></thead>"
            f"<tbody>{rows}{err_rows}</tbody>"
            f"</table>"
        )


def compute_all(
    observer: str,
    target: str,
    time: Any,
    light_time_correction: str = "NONE",
    *,
    occultors: list[str] | None = None,
    skip_errors: bool = True,
) -> PropertySnapshot:
    """Compute every auto-instantiable registered property at *time*.

    A property is considered *auto-instantiable* when it has no required
    fields beyond the standard context fields (``observer``, ``target``,
    ``light_time_correction``).  Properties that require additional
    mandatory arguments (e.g. :class:`Occultation` which needs ``front``
    and ``back``) are silently skipped from the automatic pass, but can be
    included via the *occultors* parameter.

    Parameters
    ----------
    observer:
        Observer spacecraft / instrument name (e.g. ``"JUICE_JANUS"``).
    target:
        Target body name (e.g. ``"GANYMEDE"``).
    time:
        Evaluation time — anything accepted by the properties
        (``float`` ET, ISO-8601 string, :class:`datetime.datetime`,
        :class:`pandas.Timestamp`, …).
    light_time_correction:
        SPICE aberration correction string (default ``"NONE"``).
    occultors:
        Optional list of secondary bodies for which occultation properties
        are computed.  For each body *B*, two :class:`Occultation` instances
        are evaluated:

        - *target* occulted by *B* (i.e. ``Occultation(observer, front=B, back=target)``)
        - *B* occulted by *target* (i.e. ``Occultation(observer, front=target, back=B)``)

        Pass ``occultors=["JUPITER", "IO", "EUROPA", "CALLISTO"]`` for the
        full Galilean + Jupiter set.
    skip_errors:
        When ``True`` (default) catch any evaluation error and store it in
        :attr:`PropertySnapshot.errors` rather than re-raising.  Set to
        ``False`` to let exceptions propagate.

    Returns
    -------
    PropertySnapshot
        Contains all successfully computed values, their units, the
        property instances used, and any evaluation errors.

    Examples
    --------
    ::

        snap = compute_all(
            "JUICE_JANUS", "GANYMEDE", "2032-01-01T12:00:00",
            occultors=["JUPITER", "IO", "EUROPA", "CALLISTO"],
        )
        print(snap)
        series = snap.to_series()
    """
    from spice_segmenter.core.registry import _field_info, property_registry
    from spice_segmenter.properties.occultation_types import Occultation
    from spice_segmenter.support.context import SpiceContext

    values: dict[str, Any] = {}
    units_map: dict[str, str] = {}
    props_map: dict[str, Any] = {}
    errors: dict[str, str] = {}

    with SpiceContext(
        observer=observer,
        target=target,
        light_time_correction=light_time_correction,
    ):
        for prop_name, cls in property_registry.all().items():
            # Skip helpers/wrappers that are not standalone computable properties.
            if getattr(cls, "_skip_auto_compute", False):
                continue

            required, _ctx, _opt = _field_info(cls)
            if required:
                # needs extra args we can't auto-provide — skip
                continue

            try:
                prop = cls()
            except Exception as exc:
                iid = f"{prop_name}"
                if skip_errors:
                    errors[iid] = f"instantiation failed: {exc}"
                else:
                    raise
                continue

            iid = prop.instance_id
            unit_obj = getattr(cls, "_unit", None)
            if isinstance(unit_obj, pint.Unit):
                unit_str = str(unit_obj)
            elif isinstance(unit_obj, list):
                unit_str = "[" + ", ".join(str(u) for u in unit_obj) + "]"
            else:
                unit_str = str(unit_obj) if unit_obj is not None else ""

            try:
                val = prop(time)
            except Exception as exc:
                if skip_errors:
                    errors[iid] = str(exc)
                    props_map[iid] = prop
                    units_map[iid] = unit_str
                else:
                    raise
                continue

            values[iid] = val
            units_map[iid] = unit_str
            props_map[iid] = prop

        # --- Occultation properties for secondary bodies ---
        if occultors:
            target_upper = target.upper()
            observer_upper = observer.upper()
            for body in occultors:
                body_upper = body.upper()
                if body_upper == target_upper:
                    continue  # skip self-occultation

                for front, back, label in (
                    (body_upper,   target_upper, f"{body_upper}_occults_{target_upper}"),
                    (target_upper, body_upper,   f"{target_upper}_occults_{body_upper}"),
                ):
                    try:
                        occ = Occultation(
                            observer=observer_upper,
                            front=front,
                            back=back,
                            light_time_correction=light_time_correction,
                        )
                    except Exception as exc:
                        if skip_errors:
                            errors[label] = f"instantiation failed: {exc}"
                        else:
                            raise
                        continue

                    iid = occ.instance_id
                    try:
                        val = occ(time)
                    except Exception as exc:
                        if skip_errors:
                            errors[iid] = str(exc)
                            props_map[iid] = occ
                            units_map[iid] = ""
                        else:
                            raise
                        continue

                    values[iid] = val
                    units_map[iid] = ""
                    props_map[iid] = occ

    return PropertySnapshot(
        observer=observer,
        target=target,
        time=time,
        values=values,
        units=units_map,
        properties=props_map,
        errors=errors,
    )
