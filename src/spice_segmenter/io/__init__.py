"""I/O subpackage for spice_segmenter.

Provides two complementary serialization layers:

DSL (``spice_segmenter.io.dsl``)
    Parse a human-friendly expression string into a
    :class:`~spice_segmenter.core.constraints.Constraint` and serialize it
    back to a string.

YAML I/O (``spice_segmenter.io.yaml_io``)
    Read and write constraints or property-instance lists from/to YAML files.

Quick-start
-----------
::

    from spice_segmenter.io import (
        parse,
        constraint_to_expression, constraint_to_context,
        load, loads, dump, dumps,
        load_properties, loads_properties, dump_properties, dumps_properties,
    )
"""

from spice_segmenter.io.dsl import (
    constraint_to_context,
    constraint_to_expression,
    parse,
)
from spice_segmenter.io.yaml_io import (
    dump,
    dump_properties,
    dumps,
    dumps_properties,
    load,
    load_properties,
    loads,
    loads_properties,
)

__all__ = [
    # DSL
    "parse",
    "constraint_to_expression",
    "constraint_to_context",
    # Constraint YAML
    "load",
    "loads",
    "dump",
    "dumps",
    # Property-list YAML
    "load_properties",
    "loads_properties",
    "dump_properties",
    "dumps_properties",
]
