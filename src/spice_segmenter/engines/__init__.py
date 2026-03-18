"""engines — computation dispatch layer for spice_segmenter properties.

Public API::

    from spice_segmenter.engines import get_evaluator, SpiceEngine, Evaluator

``get_evaluator()`` returns the global :class:`Evaluator` singleton which is
lazily initialised with all built-in SPICE compute functions on first access.
"""

from .evaluator import Evaluator, get_evaluator
from .spice_engine import SpiceEngine

__all__ = ["Evaluator", "SpiceEngine", "get_evaluator"]
