"""
Constraint Optimization Module

Provides systematic transformation of slow properties to faster equivalents.
For example: TargetSizeOnSensor > 5px → Distance < threshold_km

This module is entirely optional and non-invasive. Use by calling:
    optimized_constraint = optimize_constraint(constraint)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger as log

if TYPE_CHECKING:
    from spice_segmenter.constraint import ConstraintBase
    from spice_segmenter.property_base import Property


class PropertyTransformer(ABC):
    """Base class for property transformations.
    
    Subclasses should implement pattern matching and transformation logic.
    """

    @abstractmethod
    def can_transform(self, left: Property, operator: str, right: Any) -> bool:
        """Check if this transformer can handle the given constraint pattern."""

    @abstractmethod
    def transform(self, left: Property, operator: str, right: Any) -> tuple[Property, str, Any]:
        """Transform the constraint to a faster equivalent.
        
        Returns:
            (transformed_left, transformed_operator, transformed_right)
        """

    @abstractmethod
    def name(self) -> str:
        """Name of this transformer for logging."""


class TargetSizeOnSensorToDistance(PropertyTransformer):
    """Transform TargetSizeOnSensor constraints to Distance constraints.
    
    TargetSizeOnSensor > N pixels → Distance < distance_threshold
    
    This avoids expensive FOV projection calculations by using the direct
    geometric relationship: pixels = angular_size / pixel_scale
    """

    def can_transform(self, left: Property, operator: str, right: Any) -> bool:
        """Check if left is TargetSizeOnSensor and operator is comparison."""
        from spice_segmenter.trajectory_properties import TargetSizeOnSensor

        if not isinstance(left, TargetSizeOnSensor):
            return False

        # Only transform comparison operators
        return operator in ["<", ">", "<=", ">=", "!="]

    def transform(self, left: Property, operator: str, right: Any) -> tuple[Property, str, Any]:
        """Transform to Distance constraint."""

        from spice_segmenter.constant import Constant
        from spice_segmenter.trajectory_properties import Distance, TargetSizeOnSensor

        if not isinstance(left, TargetSizeOnSensor):
            raise ValueError("Left side must be TargetSizeOnSensor")

        # Extract pixel count from right side
        if isinstance(right, Constant):
            pixel_count = float(right.value)
        else:
            pixel_count = float(right)

        # Get instrument IFOV (average across all dimensions)
        ifov = np.mean(left.observer.ifov)

        # Get target radius
        target_radius = left.target.radius  # in km

        # Calculate distance threshold
        # TargetSizeOnSensor = angular_size / ifov
        # angular_size = 2 * arctan(radius / distance)
        # So: pixels = 2 * arctan(radius / distance) / ifov
        # Solving for distance: distance = radius / tan(pixels * ifov / 2)

        distance_km = target_radius / np.tan(pixel_count * ifov / 2)

        # Invert operator: > pixels means < distance
        inverted_op = {
            "<": ">",   # fewer pixels (small size) = far away
            ">": "<",   # more pixels (large size) = close
            "<=": ">=",
            ">=": "<=",
            "!=": "!=",
        }.get(operator, operator)

        # Create Distance constraint
        distance_property = Distance(observer=left.observer, target=left.target)
        distance_constant = Constant.from_value(f"{distance_km:.2f} km")

        return distance_property, inverted_op, distance_constant

    def name(self) -> str:
        return "TargetSizeOnSensor→Distance"


class AngularSizeToDistance(PropertyTransformer):
    """Transform AngularSize constraints to Distance constraints.
    
    AngularSize > angle → Distance < distance_threshold
    
    Avoids recomputing angular size by using direct distance formula.
    """

    def can_transform(self, left: Property, operator: str, right: Any) -> bool:
        """Check if left is AngularSize and operator is comparison."""
        from spice_segmenter.trajectory_properties import AngularSize

        if not isinstance(left, AngularSize):
            return False

        return operator in ["<", ">", "<=", ">=", "!="]

    def transform(self, left: Property, operator: str, right: Any) -> tuple[Property, str, Any]:
        """Transform to Distance constraint."""

        from spice_segmenter.constant import Constant
        from spice_segmenter.trajectory_properties import AngularSize, Distance

        if not isinstance(left, AngularSize):
            raise ValueError("Left side must be AngularSize")

        # Extract angle
        if isinstance(right, Constant):
            angle_val = float(right.value)
        else:
            angle_val = float(right)

        # Convert angle to radians if needed
        if hasattr(right, "unit") and right.unit:
            # Check if it's in degrees
            try:
                if "degree" in str(right.unit).lower():
                    angle_val = np.radians(angle_val)
            except:
                pass

        # Get target radius
        target_radius = left.target.radius  # in km

        # Calculate distance: angular_size = 2 * arctan(radius / distance)
        # So: distance = radius / tan(angle / 2)
        distance_km = target_radius / np.tan(angle_val / 2)

        # Invert operator
        inverted_op = {
            "<": ">",
            ">": "<",
            "<=": ">=",
            ">=": "<=",
            "!=": "!=",
        }.get(operator, operator)

        # Create Distance constraint
        distance_property = Distance(observer=left.observer, target=left.target)
        distance_constant = Constant.from_value(f"{distance_km:.2f} km")

        return distance_property, inverted_op, distance_constant

    def name(self) -> str:
        return "AngularSize→Distance"


class ConstraintOptimizer:
    """Optimizer that applies transformers to constraints.
    
    Usage:
        optimizer = ConstraintOptimizer()
        optimized = optimizer.optimize(constraint)
    """

    def __init__(self, transformers: list[PropertyTransformer] | None = None):
        """Initialize with default or custom transformers.
        
        Args:
            transformers: List of PropertyTransformer instances. 
                         If None, uses all built-in transformers.
        """
        if transformers is None:
            self.transformers = [
                TargetSizeOnSensorToDistance(),
                AngularSizeToDistance(),
            ]
        else:
            self.transformers = transformers

        self.transformations_applied: list[tuple[str, str, str]] = []

    def optimize(self, constraint: ConstraintBase) -> ConstraintBase:
        """Optimize a constraint tree by applying transformations.
        
        Args:
            constraint: The constraint to optimize
            
        Returns:
            Optimized constraint (may be the same object if no optimizations applied)
        """

        self.transformations_applied.clear()
        return self._optimize_recursive(constraint)

    def _optimize_recursive(self, constraint: ConstraintBase) -> ConstraintBase:
        """Recursively optimize constraint tree."""
        from spice_segmenter.constraint import Constraint, ConstraintBase
        from spice_segmenter.ops import Inverted, WrappedConstraint

        if isinstance(constraint, Constraint):
            # Try to transform this constraint
            left, operator, right = constraint.left, constraint.operator, constraint.right

            for transformer in self.transformers:
                try:
                    if transformer.can_transform(left, operator, right):
                        new_left, new_op, new_right = transformer.transform(left, operator, right)

                        self.transformations_applied.append((
                            str(constraint),
                            transformer.name(),
                            f"({new_left} {new_op} {new_right})",
                        ))

                        log.info(
                            f"Applied transformation: {transformer.name()}\n"
                            f"  Before: {constraint}\n"
                            f"  After: ({new_left} {new_op} {new_right})",
                        )

                        # Create optimized constraint
                        return Constraint(left=new_left, operator=new_op, right=new_right)

                except Exception as e:
                    log.debug(f"Transformer {transformer.name()} failed: {e}")
                    continue

            # No transformation matched, but recursively optimize left and right
            # Only if they are also Constraints (not Properties)
            new_left = self._optimize_recursive(left) if isinstance(left, ConstraintBase) else left
            new_right = self._optimize_recursive(right) if isinstance(right, ConstraintBase) else right

            # If either side changed, create new constraint
            if new_left is not left or new_right is not right:
                return Constraint(left=new_left, operator=operator, right=new_right)

        # Handle Inverted constraints (negations with ~)
        elif isinstance(constraint, Inverted):
            parent = constraint.parent
            optimized_parent = self._optimize_recursive(parent)

            # If parent was optimized, create new Inverted with optimized parent
            if optimized_parent is not parent:
                return Inverted(optimized_parent)

        # Handle other wrapped constraints
        elif isinstance(constraint, WrappedConstraint):
            parent = constraint.parent
            optimized_parent = self._optimize_recursive(parent)

            # If parent was optimized, create new wrapper with optimized parent
            if optimized_parent is not parent:
                # Create new wrapper of same type
                return constraint.__class__(optimized_parent)

        # Recursively optimize sub-constraints if they exist
        # (for AND/OR constraints)
        elif hasattr(constraint, "left") and hasattr(constraint, "right"):
            new_left = self._optimize_recursive(constraint.left) if isinstance(constraint.left, ConstraintBase) else constraint.left
            new_right = self._optimize_recursive(constraint.right) if isinstance(constraint.right, ConstraintBase) else constraint.right

            # If either side changed, try to create new constraint of same type
            if (new_left is not constraint.left or new_right is not constraint.right):
                try:
                    return constraint.__class__(left=new_left, operator=constraint.operator, right=new_right)
                except Exception:
                    # If that fails, just return original
                    pass

        return constraint

    def report(self) -> str:
        """Get a report of optimizations applied."""
        if not self.transformations_applied:
            return "No optimizations applied"

        lines = ["Constraint Optimizations Applied:"]
        lines.append("-" * 80)

        for original, transformer, optimized in self.transformations_applied:
            lines.append(f"🔄 {transformer}")
            lines.append(f"   From: {original}")
            lines.append(f"   To:   {optimized}")

        lines.append("-" * 80)
        return "\n".join(lines)


# Singleton optimizer instance
_default_optimizer: ConstraintOptimizer | None = None


def get_optimizer() -> ConstraintOptimizer:
    """Get the default constraint optimizer."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = ConstraintOptimizer()
    return _default_optimizer


def optimize_constraint(constraint: ConstraintBase, verbose: bool = True) -> ConstraintBase:
    """Convenience function to optimize a constraint with default optimizer.
    
    Args:
        constraint: The constraint to optimize
        verbose: If True, log applied optimizations
        
    Returns:
        Optimized constraint
        
    Examples:
        >>> from spice_segmenter import TargetSizeOnSensor
        >>> size_constraint = TargetSizeOnSensor('JUICE_JANUS', 'METIS') > '5 px'
        >>> optimized = optimize_constraint(size_constraint)
        # Now uses fast Distance check instead of expensive FOV projection
    """
    optimizer = get_optimizer()
    optimized = optimizer.optimize(constraint)

    if verbose and optimizer.transformations_applied:
        log.info(optimizer.report())

    return optimized
