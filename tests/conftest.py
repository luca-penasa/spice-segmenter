"""Project-wide pytest configuration.

Isolation guarantee
-------------------
``SpiceContext`` uses a :class:`contextvars.ContextVar` for thread / asyncio
safety, but *sequential* tests running inside the same pytest-xdist worker
process share that context variable.  A test that enters a ``SpiceContext``
without exiting (e.g. because an exception fires before ``__exit__``) would
leave ``_context_var`` pointing at a stale context, silently corrupting every
subsequent test in the same worker.

The ``isolated_spice_context`` autouse fixture (function scope) eliminates
this by resetting ``_context_var`` to the clean module-level default before
every test and un-doing that reset after the test finishes (even on failure),
so the fixture itself composes correctly with any nesting.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_spice_context():
    """Ensure every test starts (and ends) with a clean :class:`SpiceContext`.

    Implementation notes
    --------------------
    * ``ContextVar.set(value)`` returns a *token* that records what the
      variable held **before** the set.  ``ContextVar.reset(token)`` restores
      that previous value atomically — it is the official way to undo a
      ``set`` call.
    * Setting to ``_default_context`` at the start of every test guarantees
      that even if the previous test leaked a context (e.g. called
      ``__enter__`` without the matching ``__exit__``), this test sees a
      known-clean state.
    * The ``reset`` in the teardown un-does **our** set, not anything the test
      itself may have done — so nested ``with SpiceContext(...)`` blocks inside
      the code under test are still safe.
    """
    from spice_segmenter.support.context import _context_var, spice_context

    token = _context_var.set(spice_context)
    yield
    _context_var.reset(token)
