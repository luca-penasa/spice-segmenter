"""Configuration management for spice_segmenter."""

from __future__ import annotations

import contextvars

import pandas as pd
from attrs import define, field


def to_timedelta(value) -> pd.Timedelta:
    """Convert value to pd.Timedelta."""
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(seconds=value)
    if isinstance(value, str):
        return pd.Timedelta(value)
    raise ValueError(f"Cannot convert {value!r} to pd.Timedelta")


@define
class Config:
    """Configuration for the spice_segmenter module.

    Can be used as a context manager to temporarily override settings::

        with Config(solver_step="1h", start="2032-01-01", end="2035-01-01"):
            constraint.solve()        # uses default window + step from context
            lat.find_local_maxima()   # idem

    Outside any ``with`` block the module-level ``config`` singleton is active.
    Contexts nest cleanly and are thread-safe (and asyncio-safe) thanks to
    :class:`contextvars.ContextVar`.
    """

    show_progressbar: bool = field(default=False)
    solver_step: pd.Timedelta = field(
        default=pd.Timedelta(minutes=5), converter=to_timedelta
    )
    start: str | pd.Timestamp | None = field(default=None)
    end: str | pd.Timestamp | None = field(default=None)

    _token: contextvars.Token | None = field(
        default=None, init=False, repr=False
    )

    @property
    def solver_step_seconds(self) -> float:
        """Return *solver_step* as total seconds (float)."""
        return self.solver_step.total_seconds()

    def __enter__(self) -> Config:
        self._token = _config_var.set(self)
        return self

    def __exit__(self, *_args: object) -> None:
        if self._token is not None:
            _config_var.reset(self._token)
            self._token = None

    def override(self, **kwargs) -> Config:
        """Return a new :class:`Config` that copies current settings and overrides *kwargs*.

        Intended for use as a context manager::

            with config.override(solver_step="30 min"):
                constraint.solve()   # step is 30 min; all other fields unchanged

        Works equally well on the active context config::

            with get_active_config().override(start="2032-01-01", end="2033-01-01"):
                lat.find_local_maxima()
        """
        import attrs

        current = {
            a.name: getattr(self, a.name)
            for a in attrs.fields(Config)
            if a.init
        }
        current.update(kwargs)
        return Config(**current)


# Module-level singleton — always available as ``from spice_segmenter import config``.
config = Config()

# ContextVar makes nesting, thread-safety and asyncio-safety automatic.
_config_var: contextvars.ContextVar[Config] = contextvars.ContextVar(
    "_config_var",
)


def get_active_config() -> Config:
    """Return the currently active :class:`Config`.

    Inside a ``with Config(...):`` block this returns that context's config;
    outside any block it returns the module-level ``config`` singleton.
    """
    return _config_var.get(config)
