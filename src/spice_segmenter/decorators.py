from functools import wraps
from typing import Callable

import numpy as np


def vectorize(
    function: Callable | None = None,
    otypes: None | str = None,
    signature: None | str = None,
) -> None:
    """Numpy vectorization wrapper that works with instance methods."""

    def decorator(fn: Callable) -> Callable:
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    if function:
        return decorator(function)

    return decorator
