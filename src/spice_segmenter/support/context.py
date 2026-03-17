"""Thread-safe global context for default SPICE property values.

Usage
-----
Set a global default so properties can be constructed without explicit arguments::

    from spice_segmenter import (
        SpiceContext,
        Distance,
    )

    with SpiceContext(
        observer="MY_SC",
        target="ENCELADUS",
    ):
        d = Distance()  # observer/target taken from context
        d2 = Distance(
            target="TITAN"
        )  # explicit values always win

Contexts nest cleanly and are thread-safe (and asyncio-safe) thanks to
:class:`contextvars.ContextVar`.
"""

from __future__ import annotations

import contextvars

from attrs import define, field


@define
class SpiceContext:
    """Holds default values for SPICE property attributes.

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


# Module-level default context (no observer/target set, correction="NONE").
_default_context: SpiceContext = SpiceContext()

# ContextVar makes this thread-safe and asyncio-safe.
_context_var: contextvars.ContextVar[SpiceContext] = contextvars.ContextVar(
    "_spice_context_var",
)


def get_context() -> SpiceContext:
    """Return the currently active :class:`SpiceContext`."""
    return _context_var.get(_default_context)


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
