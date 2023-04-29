from functools import wraps
from typing import Any, Callable

import numpy as np


def vectorize(
    function: Callable[..., Any] | None = None,
    otypes: None | str = None,
    signature: None | str = None,
) -> Callable[..., Any]:
    """Numpy vectorization wrapper that works with instance methods."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    if function:
        return decorator(function)

    return decorator
