"""SPICE compute functions for ring-phase Boolean properties."""

from __future__ import annotations

from ...properties.ring_properties import (
    RingAnsaePhaseGreaterThan,
    RingAnsaePhaseLowerThan,
    RingAnsaePhaseWithinRange,
    is_ring_ansae_phase_angles_greater_than,
    is_ring_ansae_phase_angles_in_between,
    is_ring_ansae_phase_angles_lower_than,
)


def ring_lower_than_scalar(prop: RingAnsaePhaseLowerThan, time_et: float) -> bool:
    return bool(is_ring_ansae_phase_angles_lower_than(time_et, prop.value_deg))


def ring_greater_than_scalar(prop: RingAnsaePhaseGreaterThan, time_et: float) -> bool:
    return bool(is_ring_ansae_phase_angles_greater_than(time_et, prop.value_deg))


def ring_within_range_scalar(prop: RingAnsaePhaseWithinRange, time_et: float) -> bool:
    return bool(
        is_ring_ansae_phase_angles_in_between(time_et, prop.lower_deg, prop.upper_deg),
    )
