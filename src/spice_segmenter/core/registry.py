"""Global property registry.

All :class:`~spice_segmenter.core.property.Property` subclasses are
registered here automatically via ``Property.__init_subclass__`` as soon as
their module is imported.

Public surface
--------------
``property_registry``
    The singleton :class:`PropertyRegistry` instance.
``property_registry.register(name, cls)``
    Register a class explicitly (used internally by ``__init_subclass__``).
``property_registry[name]`` / ``property_registry.get(name)``
    Look up a class by its ``_name`` string.
``property_registry.all()``
    Return a snapshot ``dict`` of the full registry.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Iterator

import attrs

if TYPE_CHECKING:
    from spice_segmenter.core.property import Property

# Names of the three context-provided fields — shown separately in the repr.
_CONTEXT_FIELDS = {"observer", "target", "light_time_correction"}


def _field_info(cls: type["Property"]) -> tuple[list[str], list[str], list[str]]:
    """Return (required, context, optional) parameter name lists for *cls*.

    - **required**: ``init=True`` fields with ``default=NOTHING``
    - **context**: ``init=True`` fields whose name is in ``_CONTEXT_FIELDS``
    - **optional**: ``init=True`` fields with any explicit default/factory,
      excluding context fields
    """
    required: list[str] = []
    context: list[str] = []
    optional: list[str] = []

    try:
        fields = attrs.fields(cls)
    except attrs.exceptions.NotAnAttrsClassError:
        return required, context, optional

    for f in fields:
        if not f.init:
            continue
        if f.name.startswith("_"):
            continue
        if f.name in _CONTEXT_FIELDS:
            context.append(f.name)
        elif f.default is attrs.NOTHING:
            required.append(f.name)
        else:
            optional.append(f.name)

    return required, context, optional


def _unit_str(cls: type["Property"]) -> str:
    unit = getattr(cls, "_unit", None)
    if unit is None:
        return ""
    if isinstance(unit, list):
        return "[" + ", ".join(str(u) for u in unit) + "]"
    return str(unit)


def _type_str(cls: type["Property"]) -> str:
    from spice_segmenter.core.property import PropertyTypes
    ptype = getattr(cls, "_type", PropertyTypes.SCALAR)
    return ptype.name.capitalize()


class PropertyRegistry:
    """Centralised registry of all :class:`~spice_segmenter.core.property.Property` subclasses.

    Instances behave like a read-only mapping: iterate with ``for name in registry``,
    look up with ``registry[name]``, check membership with ``"distance" in registry``.
    """

    def __init__(self) -> None:
        self._store: dict[str, type["Property"]] = {}

    # ------------------------------------------------------------------
    # Mutation (internal use)
    # ------------------------------------------------------------------

    def register(self, name: str, cls: type["Property"]) -> type["Property"]:
        """Register *cls* under *name*.  Called by ``Property.__init_subclass__``.

        Three cases are handled:

        1. **No existing entry** — register and return.
        2. **Same class object** — no-op (idempotent re-import).
        3. **attrs slots-rewrite** — ``attrs`` with ``__slots__`` replaces a
           class by re-calling ``type(cls.__name__, cls.__bases__, ...)`` which
           re-fires ``__init_subclass__``.  The pre-slots class (first call)
           has no ``__attrs_attrs__`` in its own ``__dict__``; the post-slots
           class (second call) does.  We update the registry to the new object.
        4. **Genuine conflict** — two separate class definitions claim the same
           property name.  This causes ``type(a) is type(b)`` to fail for
           values produced by different code paths (e.g. a class-based property
           and a ``@property_function``-generated class with the same name), and
           leads to intermittent test failures depending on import order.  A
           ``ValueError`` is raised so the problem surfaces immediately.
        """
        existing = self._store.get(name)
        if existing is None:
            # Case 1: first registration.
            self._store[name] = cls
        elif existing is cls:
            # Case 2: idempotent re-import, nothing to do.
            pass
        elif (
            "__attrs_attrs__" not in existing.__dict__
            and "__attrs_attrs__" in cls.__dict__
            and existing.__qualname__ == cls.__qualname__
            and existing.__module__ == cls.__module__
        ):
            # Case 3: attrs slots-rewrite — update to the authoritative object.
            self._store[name] = cls
        else:
            # Case 4: genuine conflict between two different classes.
            raise ValueError(
                f"Property name {name!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__} "
                f"and cannot be overwritten by "
                f"{cls.__module__}.{cls.__qualname__}. "
                f"Remove the duplicate @property_function / _name declaration, "
                f"or import the existing class instead of redefining it."
            )
        return cls

    def register_or_skip(self, name: str, cls: type["Property"]) -> type["Property"]:
        """Like :meth:`register` but silently skips on genuine conflicts.

        Used by ``@property_function`` and other alternative/functional
        definitions that are intentional secondary implementations of an
        already-registered property.  The first registration wins; subsequent
        ones are ignored rather than raising.
        """
        try:
            self.register(name, cls)
        except ValueError:
            pass  # Already owned by a different class — the first one wins.
        return self._store.get(name, cls)

    # ------------------------------------------------------------------
    # Mapping-like read API
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> type["Property"]:
        try:
            return self._store[name]
        except KeyError:
            available = ", ".join(sorted(self._store))
            raise KeyError(
                f"No property named {name!r} is registered. "
                f"Available: {available}"
            ) from None

    def get(self, name: str, default: type["Property"] | None = None) -> type["Property"] | None:
        """Return the class for *name*, or *default* if not found."""
        return self._store.get(name, default)

    def __contains__(self, name: object) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def all(self) -> dict[str, type["Property"]]:
        """Return a snapshot copy of the full registry."""
        return dict(self._store)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._store:
            return "PropertyRegistry (empty)"

        col_name = max(len(n) for n in self._store)
        col_unit = max(len(_unit_str(c)) for c in self._store.values()) or 4
        col_type = max(len(_type_str(c)) for c in self._store.values()) or 4
        col_name = max(col_name, 4)

        header = (
            f"{'Name':<{col_name}}  {'Type':<{col_type}}  {'Unit':<{col_unit}}  "
            f"Required  Context          Optional"
        )
        sep = "-" * len(header)
        lines = [f"PropertyRegistry ({len(self._store)} properties)", sep, header, sep]

        for name in sorted(self._store):
            cls = self._store[name]
            req, ctx, opt = _field_info(cls)
            lines.append(
                f"{name:<{col_name}}  "
                f"{_type_str(cls):<{col_type}}  "
                f"{_unit_str(cls):<{col_unit}}  "
                f"{', '.join(req) or '-':<8}  "
                f"{', '.join(ctx) or '-':<16}  "
                f"{', '.join(opt) or '-'}"
            )

        lines.append(sep)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML table for Jupyter notebooks (theme-aware via CSS)."""
        rows_html = ""
        for name in sorted(self._store):
            cls = self._store[name]
            req, ctx, opt = _field_info(cls)

            def _badge(params: list[str], kind: str) -> str:
                if not params:
                    return "<em class='pr-none'>—</em>"
                return " ".join(
                    f"<code class='pr-badge pr-{kind}'>{p}</code>"
                    for p in params
                )

            rows_html += (
                f"<tr>"
                f"<td><strong>{name}</strong></td>"
                f"<td><span class='pr-badge pr-type pr-type-{_type_str(cls).lower()}'>{_type_str(cls)}</span></td>"
                f"<td><code>{_unit_str(cls)}</code></td>"
                f"<td>{_badge(req, 'required')}</td>"
                f"<td>{_badge(ctx, 'context')}</td>"
                f"<td>{_badge(opt, 'optional')}</td>"
                f"</tr>"
            )

        return textwrap.dedent(f"""
            <style>
              .pr-table {{
                border-collapse: collapse;
                font-family: monospace;
              }}
              .pr-table th, .pr-table td {{
                text-align: left;
                padding: 4px 8px;
              }}
              .pr-table thead tr {{
                border-bottom: 2px solid currentColor;
                opacity: 0.9;
              }}
              .pr-none {{
                opacity: 0.4;
              }}
              .pr-badge {{
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 0.85em;
                border: 1px solid currentColor;
                opacity: 0.85;
              }}
              .pr-type-scalar   {{ border-color: orchid;  color: orchid; }}
              .pr-type-vector   {{ border-color: darkorange; color: darkorange; }}
              .pr-type-boolean  {{ border-color: steelblue;  color: steelblue; }}
              .pr-type-discrete {{ border-color: slategray;  color: slategray; }}
              .pr-required  {{ border-color: tomato;       color: tomato; }}
              .pr-context   {{ border-color: cornflowerblue; color: cornflowerblue; }}
              .pr-optional  {{ border-color: mediumseagreen; color: mediumseagreen; }}
              .pr-th-required {{ color: tomato; }}
              .pr-th-context  {{ color: cornflowerblue; }}
              .pr-th-optional {{ color: mediumseagreen; }}
            </style>
            <table class='pr-table'>
              <thead>
                <tr>
                  <th>Name</th>
                  <th><span class='pr-badge pr-type'>Type</span></th>
                  <th>Unit</th>
                  <th><span class='pr-badge pr-required'>Required</span></th>
                  <th><span class='pr-badge pr-context'>Context</span></th>
                  <th><span class='pr-badge pr-optional'>Optional</span></th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
        """).strip()


# ---------------------------------------------------------------------------
# Singleton — the single source of truth used throughout the package.
# ---------------------------------------------------------------------------
property_registry: PropertyRegistry = PropertyRegistry()


# ---------------------------------------------------------------------------
# Module-level shims kept for backwards compatibility and for import in
# core/__init__.py.
# ---------------------------------------------------------------------------

def register(name: str, cls: type["Property"]) -> type["Property"]:
    """Register *cls* under *name* in the global registry."""
    return property_registry.register(name, cls)


def get(name: str) -> type["Property"]:
    """Return the property class registered under *name*."""
    return property_registry[name]


def all() -> dict[str, type["Property"]]:  # noqa: A001
    """Return a snapshot copy of the full registry."""
    return property_registry.all()
