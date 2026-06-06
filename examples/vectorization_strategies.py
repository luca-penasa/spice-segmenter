"""
Example: Using vectorized computation strategies.

This example shows three approaches to handling vectorized property computations:
1. Manual vectorization in the function
2. Backend-based vectorization dispatch
3. Specialized vectorized backend
"""

import numpy as np
from spice_segmenter.properties.functional import Distance
from spice_segmenter.backends import register_backend
from spice_segmenter.backends.vectorized_backend import VectorizedSpiceBackend


# =============================================================================
# Approach 1: Current default - automatic vectorization
# =============================================================================
def example_1_automatic():
    """The @property_function decorator handles vectorization automatically."""
    
    # Create property (uses @vectorize decorator automatically)
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    # Works with scalars
    scalar_result = dist(0.0)
    print(f"Scalar: {scalar_result}")
    
    # Works with arrays (automatically vectorized)
    times = np.linspace(0, 86400, 100)
    array_results = dist(times)
    print(f"Array: {array_results.shape}")


# =============================================================================
# Approach 2: Register optimized vectorized version with SPICE backend
# =============================================================================
def example_2_backend_registration():
    """Register a custom vectorized computation."""
    from spice_segmenter.backends.spice_backend import register_spice_computation
    
    def optimized_distance(time, observer, target, light_time_correction="NONE"):
        """Custom distance with optimized vectorization."""
        from spice_segmenter.support.spice_utilities import et
        import spiceypy
        
        times_et = et(time)
        
        # Scalar path
        if np.isscalar(times_et):
            pos, _ = spiceypy.spkpos(
                target.name, times_et, observer.frame.name,
                light_time_correction, observer.name
            )
            return spiceypy.vnorm(pos)
        
        # Optimized vectorized path
        # (In real code: use parallel computation, batch APIs, caching, etc.)
        results = np.empty(len(times_et))
        for i, t in enumerate(times_et):
            pos, _ = spiceypy.spkpos(
                target.name, t, observer.frame.name,
                light_time_correction, observer.name
            )
            results[i] = spiceypy.vnorm(pos)
        
        return results
    
    # Register this optimized version
    register_spice_computation("distance", optimized_distance)
    
    # Now Distance property will use this implementation
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    times = np.linspace(0, 86400, 100)
    results = dist(times)
    print(f"Custom vectorized: {results.shape}")


# =============================================================================
# Approach 3: Specialized backend with priority
# =============================================================================
def example_3_vectorized_backend():
    """Use a specialized backend for array computations."""
    
    # Register vectorized backend with higher priority
    vectorized_backend = VectorizedSpiceBackend(priority=10)
    register_backend(vectorized_backend)
    
    # Now array computations will use the specialized backend
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    # Scalar still works (falls back to regular SPICE backend)
    scalar = dist(0.0)
    print(f"Scalar (regular backend): {scalar}")
    
    # Array uses vectorized backend
    times = np.linspace(0, 86400, 100)
    array = dist(times)
    print(f"Array (vectorized backend): {array.shape}")


# =============================================================================
# Approach 4: Property-specific optimization
# =============================================================================
def example_4_custom_property():
    """Create a property with custom vectorization logic."""
    from spice_segmenter.core.property_decorator import property_function
    from spice_segmenter.core.property import PropertyTypes
    import spiceypy
    from spice_segmenter.support.spice_utilities import et
    
    @property_function(
        name="optimized_distance",
        unit="km",
        property_type=PropertyTypes.SCALAR,
        vectorized=False  # We handle it ourselves
    )
    def optimized_distance(time, observer, target, light_time_correction="NONE"):
        """Distance with custom vectorization strategy."""
        times_et = et(time)
        
        # Check if vectorized
        is_array = not np.isscalar(times_et)
        
        if is_array:
            # Use optimized batch computation
            print("Using optimized batch computation")
            results = np.empty(len(times_et))
            # Could use parallel processing here
            for i, t in enumerate(times_et):
                pos, _ = spiceypy.spkpos(
                    target.name, t, observer.frame.name,
                    light_time_correction, observer.name
                )
                results[i] = spiceypy.vnorm(pos)
            return results
        else:
            # Use standard scalar computation
            print("Using scalar computation")
            pos, _ = spiceypy.spkpos(
                target.name, times_et, observer.frame.name,
                light_time_correction, observer.name
            )
            return spiceypy.vnorm(pos)
    
    # Use the property
    dist = optimized_distance("JUICE_JANUS", "GANYMEDE")
    
    scalar = dist(0.0)
    array = dist(np.linspace(0, 86400, 100))
    
    return scalar, array


# =============================================================================
# Performance comparison
# =============================================================================
def compare_approaches():
    """Compare performance of different vectorization approaches."""
    import time
    
    times = np.linspace(0, 86400, 1000)
    dist = Distance("JUICE_JANUS", "GANYMEDE")
    
    # Time the computation
    start = time.time()
    results = dist(times)
    elapsed = time.time() - start
    
    print(f"Computed {len(times)} distances in {elapsed:.3f}s")
    print(f"Average: {elapsed/len(times)*1000:.3f}ms per point")
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Vectorization Strategy Examples")
    print("=" * 70)
    
    print("\n1. Automatic vectorization (default):")
    example_1_automatic()
    
    print("\n2. Backend registration:")
    # example_2_backend_registration()
    
    print("\n3. Vectorized backend:")
    # example_3_vectorized_backend()
    
    print("\n4. Custom property:")
    # example_4_custom_property()
