"""Configuration management for spice_segmenter."""

import pandas as pd
from attrs import define, field


def to_seconds(value):
    """Convert value to seconds if needed."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return pd.Timedelta(value).total_seconds()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    raise ValueError(f"Cannot convert {value} to seconds")


@define
class Config:
    """Configuration for the spice_segmenter module."""

    show_progressbar: bool = field(default=False)
    solver_step: float = field(default=5 * 60, converter=to_seconds)


config = Config()
