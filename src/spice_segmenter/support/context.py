"""Thread-safe global context for default SPICE property values.

Usage
-----
Three ways to set defaults, ordered from most permanent to most scoped:

**1. Set-and-forget (singleton mutation)**::

    from spice_segmenter import spice_context

    spice_context.target = "JUPITER"
    spice_context.observer = "JUICE_JANUS"
    d = Distance()   # picks up both automatically

**2. Scoped override (inherits singleton, overrides selected fields)**::

    with spice_context.override(target="GANYMEDE"):
        d = Distance()   # GANYMEDE; observer still from singleton
    # after block: Distance() → JUPITER again

**3. Full explicit context**::

    with SpiceContext(observer="MY_SC", target="ENCELADUS"):
        d = Distance()   # fully specified, nests over any outer context

Contexts nest cleanly and are thread-safe (and asyncio-safe) thanks to
:class:`contextvars.ContextVar`.  Direct mutation of ``spice_context`` is
intended for single-threaded scripts and notebooks only; use ``override()``
inside threads for per-thread isolation.
"""

from __future__ import annotations

import contextvars

import attrs
from attrs import define, field


@define
class SpiceContext:
    """Holds default values for SPICE property attributes.

    Can be used as a context manager to temporarily override settings::

        with SpiceContext(observer="MY_SC", target="ENCELADUS"):
            d = Distance()   # uses MY_SC / ENCELADUS

    Or mutate the module-level ``spice_context`` singleton directly for a
    permanent session-wide default::

        spice_context.target = "JUPITER"

    Parameters
    ----------
    observer:
        Default observer (spacecraft / instrument name or :class:`SpiceRef`).
    target:
        Default target body (name or :class:`SpiceRef`).
    light_time_correction:
        Default aberration/light-time correction string (default ``"NONE"``).
    """

    target: str | None = field(default=None)
    observer: str | None = field(default="JUICE_JANUS")
    light_time_correction: str = field(default="NONE")
    _token: contextvars.Token[SpiceContext] | None = field(
        default=None, init=False, repr=False,
    )

    def __enter__(self) -> SpiceContext:
        self._token = _context_var.set(self)
        return self

    def __exit__(self, *_args: object) -> None:
        if self._token is not None:
            _context_var.reset(self._token)
            self._token = None

    def override(self, **kwargs) -> SpiceContext:
        """Return a new :class:`SpiceContext` copying current fields and overriding *kwargs*.

        Intended for use as a context manager::

            # Temporarily change only the target; observer is inherited
            with spice_context.override(target="GANYMEDE"):
                d = Distance()

            # Works equally well inside an existing context
            with get_active_context().override(light_time_correction="LT+S"):
                ...
        """
        current = {
            a.name: getattr(self, a.name)
            for a in attrs.fields(SpiceContext)
            if a.init
        }
        current.update(kwargs)
        return SpiceContext(**current)


# ---------------------------------------------------------------------------
# Module-level singleton — mutable for set-and-forget session defaults.
# ContextVar makes nesting, thread-safety and asyncio-safety automatic.
# ---------------------------------------------------------------------------

#: Module-level singleton.  Mutate directly for session-wide defaults::
#:
#:     spice_context.target = "JUPITER"
#:
#: Use ``override()`` for scoped temporary changes.
spice_context: SpiceContext = SpiceContext()

_context_var: contextvars.ContextVar[SpiceContext] = contextvars.ContextVar(
    "_spice_context_var",
)


def get_active_context() -> SpiceContext:
    """Return the currently active :class:`SpiceContext`.

    Inside a ``with SpiceContext(...):`` block this returns that context;
    outside any block it returns the module-level ``spice_context`` singleton.
    """
    return _context_var.get(spice_context)


# Backward-compatible alias.
get_context = get_active_context


# ---------------------------------------------------------------------------
# Getter helpers — passed as *factory* arguments to attrs fields
# ---------------------------------------------------------------------------


def get_current_observer() -> str | None:
    """Return the observer set in the current context.

    Raises
    ------
    RuntimeError
        If no observer has been set in the active context and none was
        supplied explicitly to the property constructor.
    """
    value = get_context().observer
    if value is None:
        msg = (
            "No default observer is set in the current SpiceContext. "
            "Either pass 'observer=...' explicitly or activate a context:\n\n"
            "    with SpiceContext(observer='MY_SC', target='ENCELADUS'):\n"
            "        ..."
        )
        raise RuntimeError(msg)

    return value


def get_current_target() -> str | None:
    """Return the target set in the current context.

    Raises
    ------
    RuntimeError
        If no target has been set in the active context and none was
        supplied explicitly to the property constructor.
    """
    value = get_context().target
    if value is None:
        msg = (
            "No default target is set in the current SpiceContext. "
            "Either pass 'target=...' explicitly or activate a context:\n\n"
            "    with SpiceContext(observer='MY_SC', target='ENCELADUS'):\n"
            "        ..."
        )

        raise RuntimeError(msg)
    return value


def get_current_light_time_correction() -> str:
    """Return the light-time correction string from the current context."""
    return get_context().light_time_correction
