"""
Helper for conditional segmentation of a trajectory.

Warning
-------

This is just a stub, that might never see the light
"""

import sys

from loguru import logger

# entirely disables logging for the spice_segmenter module
# logger.disable("spice_segmenter")
logger.remove()

logger.add(sys.stderr, level="WARNING")
