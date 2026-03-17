"""Plotting utilities for visualizing SPICE properties over time."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from matplotlib.figure import Figure

from ..core.property import Property
from ..support.time_types import TIMES_TYPES

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class _CallableWrapper:
    """Wraps a generic callable to make it compatible with Property interface."""
    
    def __init__(self, func: Callable, name: str | None = None):
        self._func = func
        self._name = name or getattr(func, '__name__', 'callable')
        self._unit = pint.Unit('')  # No unit for generic callables
    
    def __call__(self, time: TIMES_TYPES):
        """Evaluate the callable at the given time(s)."""
        result = self._func(time)
        # Handle both scalar and array results
        if isinstance(result, (list, tuple, np.ndarray)):
            return np.array(result)
        return result
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def unit(self) -> pint.Unit:
        return self._unit
    
    def has_unit(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return self._name


def _resolve_time_range(time_range, resolution: str) -> pd.DatetimeIndex:
    """Convert *time_range* (or the active Config default window) to a DatetimeIndex."""
    from ..support.config import get_active_config

    if time_range is None:
        cfg = get_active_config()
        if cfg.start is None or cfg.end is None:
            raise RuntimeError(
                "plot called without a time_range and no default window is set. "
                "Either pass time_range explicitly or activate a context:\n\n"
                "    with Config(start='2032-01-01', end='2035-01-01'):\n"
                "        plot_property(prop)\n"
            )
        time_range = (cfg.start, cfg.end)

    if isinstance(time_range, pd.DatetimeIndex):
        return time_range
    if hasattr(time_range, 'start') and hasattr(time_range, 'end'):
        return pd.date_range(pd.Timestamp(time_range.start), pd.Timestamp(time_range.end), freq=resolution)
    start_time, end_time = time_range
    return pd.date_range(pd.Timestamp(start_time), pd.Timestamp(end_time), freq=resolution)


def plot_property(
    property_obj: Property,
    time_range: tuple[TIMES_TYPES, TIMES_TYPES] | pd.DatetimeIndex | None = None,
    resolution: str = "1min",
    ax: Axes | None = None,
    label: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot a single property over a time range.
    
    Args:
        property_obj: Property instance to evaluate and plot
        time_range: Either (start, end) tuple, object with start/end attributes, or pd.DatetimeIndex.
                   If omitted, the default window from the active Config context is used.
        resolution: Time resolution for sampling (e.g., '1min', '10s', '1h')
        ax: Matplotlib axes to plot on (creates new figure if None)
        label: Label for the line (uses property name if None)
        **kwargs: Additional keyword arguments passed to plt.plot()
    
    Returns:
        Tuple of (figure, axes) objects
    
    Example:
        >>> from spice_segmenter import Distance
        >>> dist = Distance('JUICE_JANUS', 'GANYMEDE')
        >>> fig, ax = plot_property(dist, (start_time, end_time), resolution='5min')
        >>> plt.show()
    """
    times = _resolve_time_range(time_range, resolution)
    
    # Evaluate property at all times
    values = property_obj(times)
    
    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # Handle numpy arrays (vectorized output)
    if isinstance(values, np.ndarray):
        # Check if this is a 2D array (multiple values per time)
        if values.ndim == 2:
            # Plot each column separately
            n_channels = values.shape[1]
            for j in range(n_channels):
                channel_label = f"{label or str(property_obj)} [{j}]"
                ax.plot(times, values[:, j], label=channel_label, **kwargs)
        else:
            # 1D array, flatten and plot
            values = values.flatten()
            plot_label = label if label else str(property_obj)
            ax.plot(times, values, label=plot_label, **kwargs)
    else:
        # Scalar values
        plot_label = label if label else str(property_obj)
        ax.plot(times, values, label=plot_label, **kwargs)
    
    # Format
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ylabel = f"{property_obj.name}"
    if hasattr(property_obj, 'unit') and property_obj.has_unit():
        ylabel += f" [{property_obj.unit}]"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    
    return fig, ax


def plot_properties(
    properties: list[Property],
    time_range: tuple[TIMES_TYPES, TIMES_TYPES] | pd.DatetimeIndex | None = None,
    resolution: str = "1min",
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
    sharex: bool = True,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Plot multiple properties as subplots over a time range.
    
    Args:
        properties: List of Property instances to evaluate and plot
        time_range: Either (start, end) tuple, object with start/end attributes, or pd.DatetimeIndex.
                   If omitted, the default window from the active Config context is used.
        resolution: Time resolution for sampling (e.g., '1min', '10s', '1h')
        labels: Optional list of labels (uses property names if None)
        figsize: Figure size as (width, height)
        sharex: Whether to share x-axis across subplots
        **kwargs: Additional keyword arguments passed to plt.plot()
    
    Returns:
        Tuple of (figure, list of axes) objects
    
    Example:
        >>> from spice_segmenter import Distance, PhaseAngle
        >>> dist = Distance('JUICE_JANUS', 'GANYMEDE')
        >>> phase = PhaseAngle('JUICE_JANUS', 'GANYMEDE')
        >>> fig, axes = plot_properties([dist, phase], (start_time, end_time))
        >>> plt.show()
    """
    times = _resolve_time_range(time_range, resolution)
    
    # Create subplots
    n_props = len(properties)
    fig, axes = plt.subplots(n_props, 1, figsize=figsize, sharex=sharex)
    
    # Handle single property case (axes is not a list)
    if n_props == 1:
        axes = [axes]
    
    # Plot each property
    for i, prop in enumerate(properties):
        label = labels[i] if labels and i < len(labels) else None
        values = prop(times)
        
        # Handle numpy arrays
        if isinstance(values, np.ndarray):
            # Check if this is a 2D array (multiple values per time)
            if values.ndim == 2:
                # Plot each column separately
                n_channels = values.shape[1]
                for j in range(n_channels):
                    channel_label = f"{label or str(prop)} [{j}]" if label else f"{str(prop)} [{j}]"
                    axes[i].plot(times, values[:, j], label=channel_label, **kwargs)
            else:
                # 1D array, flatten and plot normally
                values = values.flatten()
                plot_label = label if label else str(prop)
                axes[i].plot(times, values, label=plot_label, **kwargs)
        else:
            # Scalar or other type
            plot_label = label if label else str(prop)
            axes[i].plot(times, values, label=plot_label, **kwargs)
        
        # Format
        ylabel = f"{prop.name}"
        if hasattr(prop, 'unit') and prop.has_unit():
            ylabel += f" [{prop.unit}]"
        axes[i].set_ylabel(ylabel, fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Only label x-axis on bottom plot
    axes[-1].set_xlabel("Time (UTC)", fontsize=11)
    
    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    fig.tight_layout()
    
    return fig, axes


def plot_properties_overlaid(
    properties: list[Property],
    time_range: tuple[TIMES_TYPES, TIMES_TYPES] | pd.DatetimeIndex | None = None,
    resolution: str = "1min",
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (12, 6),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot multiple properties on the same axes over a time range.
    
    Useful when properties have the same units or when comparing trends.
    
    Args:
        properties: List of Property instances to evaluate and plot
        time_range: Either (start, end) tuple, object with start/end attributes, or pd.DatetimeIndex.
                   If omitted, the default window from the active Config context is used.
        resolution: Time resolution for sampling (e.g., '1min', '10s', '1h')
        labels: Optional list of labels (uses property names if None)
        figsize: Figure size as (width, height)
        **kwargs: Additional keyword arguments passed to plt.plot()
    
    Returns:
        Tuple of (figure, axes) objects
    
    Example:
        >>> from spice_segmenter import Distance
        >>> dist_ganymede = Distance('JUICE_JANUS', 'GANYMEDE')
        >>> dist_europa = Distance('JUICE_JANUS', 'EUROPA')
        >>> fig, ax = plot_properties_overlaid(
        ...     [dist_ganymede, dist_europa],
        ...     (start_time, end_time),
        ...     labels=['Ganymede', 'Europa']
        ... )
        >>> plt.show()
    """
    # Parse time range
    times = _resolve_time_range(time_range, resolution)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each property
    for i, prop in enumerate(properties):
        label = labels[i] if labels and i < len(labels) else str(prop)
        values = prop(times)
        
        # Handle numpy arrays
        if isinstance(values, np.ndarray):
            values = values.flatten()
        
        ax.plot(times, values, label=label, **kwargs)
    
    # Format
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)  # Generic ylabel since multiple properties
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    
    return fig, ax


def quick_plot(
    properties: list[str | Property | Callable] | None = None,
    time_range: tuple[TIMES_TYPES, TIMES_TYPES] | None = None,
    target: str = "JUPITER",
    observer: str = "JUICE_JANUS",
    resolution: str = "1min",
    figsize: tuple[float, float] = (6, 5),
    highlight: tuple | list[tuple] | object | list[object] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Quick plot of common observation properties for a target.
    
    Convenience function that creates and plots standard properties without
    needing to instantiate them manually.
    
    Args:
        time_range: Either (start, end) tuple or object with start/end attributes
        target: Target body name (default: 'JUPITER')
        observer: Observer/instrument name (default: 'JUICE_JANUS')
        resolution: Time resolution for sampling (e.g., '1min', '10s', '1h')
        properties: List of property names (strings), Property instances, or callables
                   Callables should take a single time value and return one or more values
                   (default: ['distance', 'phase_angle', 'angular_size'])
        figsize: Figure size as (width, height)
        highlight: Optional time ranges to highlight with axvspan. Can be:
                  - Single (start, end) tuple
                  - List of (start, end) tuples
                  - Single object with start/end attributes
                  - List of objects with start/end attributes
    
    Returns:
        Tuple of (figure, list of axes) objects
    
    Example:
        >>> # Using property names
        >>> fig, axes = quick_plot((start_time, end_time), target='GANYMEDE')
        >>> 
        >>> # Using Property instances
        >>> from spice_segmenter import Distance, PhaseAngle
        >>> dist = Distance('JUICE_JANUS', 'EUROPA')
        >>> phase = PhaseAngle('JUICE_JANUS', 'EUROPA')
        >>> fig, axes = quick_plot((start_time, end_time), properties=[dist, phase])
        >>>
        >>> # Using callables
        >>> def my_metric(t):
        ...     return abs(phase_angle(t) - 90)
        >>> fig, axes = quick_plot((start_time, end_time), properties=[dist, my_metric])
        >>>
        >>> # Mix of names, instances, and callables
        >>> fig, axes = quick_plot((start_time, end_time), properties=['angular_size', dist, my_metric])
        >>>
        >>> # With highlighted regions
        >>> fig, axes = quick_plot((start_time, end_time), highlight=[('2024-01-01', '2024-01-02')])
        >>> plt.show()
    """
    from ..collections.property_collections import TargetProperties
    
    # Default properties to plot
    if properties is None:
        properties = ['distance', 'phase_angle', 'angular_size']
    
    # Create property collection
    target_props = TargetProperties(target=target, observer=observer)
    
    # Get property instances or wrap callables
    prop_objects = []
    for prop in properties:
        if isinstance(prop, Property):
            # Already a Property instance, use it directly
            prop_objects.append(prop)
        elif isinstance(prop, str):
            # Property name, look it up in target_props
            if hasattr(target_props, prop):
                prop_objects.append(getattr(target_props, prop))
            else:
                raise ValueError(f"Property '{prop}' not found in TargetProperties")
        elif callable(prop):
            # Generic callable - wrap it as a Property-like object
            prop_objects.append(_CallableWrapper(prop))
        else:
            raise TypeError(f"Property must be a string, Property instance, or callable, got {type(prop)}")
    
    # Plot
    fig, axes = plot_properties(prop_objects, time_range, resolution, figsize=figsize)
    
    # Add highlighted regions if provided
    if highlight is not None:
        # Normalize to list of ranges
        ranges_to_highlight = []
        
        # Check if it's a single tuple (start, end)
        if isinstance(highlight, tuple) and len(highlight) == 2:
            # Could be single (start, end) or list of tuples
            # Check if first element looks like a time value
            try:
                pd.Timestamp(highlight[0])
                # It's a single (start, end) tuple
                ranges_to_highlight = [highlight]
            except (ValueError, TypeError):
                # It's a list/tuple of ranges
                ranges_to_highlight = list(highlight)
        # Check if it's iterable and contains items with start/end attributes
        elif hasattr(highlight, '__iter__') and not isinstance(highlight, (str, dict)):
            # Try to iterate and check if items have start/end
            try:
                items = list(highlight)
                if items and hasattr(items[0], 'start') and hasattr(items[0], 'end'):
                    # It's an iterable of objects with start/end
                    ranges_to_highlight = items
                elif items:
                    # It's a regular list/iterable (assume list of tuples)
                    ranges_to_highlight = items
                else:
                    # Empty iterable
                    ranges_to_highlight = []
            except (TypeError, AttributeError):
                # Not iterable or iteration failed, check if it's a single object
                if hasattr(highlight, 'start') and hasattr(highlight, 'end'):
                    ranges_to_highlight = [highlight]
                else:
                    raise TypeError(
                        f"highlight must be a (start, end) tuple, list of tuples, "
                        f"iterable of objects with start/end attributes, or single object with start/end. Got {type(highlight)}"
                    )
        # Check if it's a single object with start/end attributes
        elif hasattr(highlight, 'start') and hasattr(highlight, 'end'):
            ranges_to_highlight = [highlight]
        else:
            raise TypeError(
                f"highlight must be a (start, end) tuple, list of tuples, "
                f"object with start/end attributes, or list of such objects. Got {type(highlight)}"
            )
        
        # Add axvspan to all axes
        for range_spec in ranges_to_highlight:
            # Extract start and end
            if isinstance(range_spec, tuple):
                start, end = range_spec
            elif hasattr(range_spec, 'start') and hasattr(range_spec, 'end'):
                start = range_spec.start
                end = range_spec.end
            else:
                raise TypeError(
                    f"Each highlight range must be a (start, end) tuple or "
                    f"object with start/end attributes. Got {type(range_spec)}"
                )
            
            # Convert to timestamps
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            
            # Add to all axes
            for ax in axes:
                ax.axvspan(start_ts, end_ts, alpha=0.2, color='yellow', zorder=-1)
    
    return fig, axes


__all__ = [
    "plot_property",
    "plot_properties",
    "plot_properties_overlaid",
    "quick_plot",
]
