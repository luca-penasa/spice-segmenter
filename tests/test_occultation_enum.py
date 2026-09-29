from spice_segmenter.properties.occultation_types import OccultationTypes


def test_enum() -> None:
    assert OccultationTypes.ANNULAR == OccultationTypes.ANNULAR
    assert OccultationTypes.FULL == OccultationTypes.FULL
    assert OccultationTypes.NONE == OccultationTypes.NONE
    assert OccultationTypes.PARTIAL == OccultationTypes.PARTIAL

    assert OccultationTypes.ANNULAR != OccultationTypes.FULL
    assert OccultationTypes.ANNULAR != OccultationTypes.NONE
    assert OccultationTypes.ANNULAR != OccultationTypes.PARTIAL

    assert OccultationTypes.FULL != OccultationTypes.ANNULAR
    assert OccultationTypes.FULL != OccultationTypes.NONE
    assert OccultationTypes.FULL != OccultationTypes.PARTIAL

    assert OccultationTypes.NONE != OccultationTypes.ANNULAR
    assert OccultationTypes.NONE != OccultationTypes.FULL
    assert OccultationTypes.NONE != OccultationTypes.PARTIAL

    assert OccultationTypes.ANY == OccultationTypes.ANY
    assert OccultationTypes.ANY == OccultationTypes.PARTIAL
    assert OccultationTypes.ANY == OccultationTypes.FULL
    assert OccultationTypes.ANY == OccultationTypes.ANNULAR
    assert OccultationTypes.ANY != OccultationTypes.NONE

    assert OccultationTypes.PARTIAL == OccultationTypes.ANY
    assert OccultationTypes.FULL == OccultationTypes.ANY
    assert OccultationTypes.ANNULAR == OccultationTypes.ANY
    assert OccultationTypes.NONE != OccultationTypes.ANY


def test_vector_evaluation_matches_scalar() -> None:
    """Vectorised occultation (cyice.occult_v) gives the same states as the scalar path."""
    import numpy as np
    import spiceypy

    from spice_segmenter.properties.occultation_types import Occultation
    from tests import tour_config  # noqa: F401 — loads SPICE kernels as a side-effect

    times = np.arange(
        spiceypy.str2et("2033-11-22"), spiceypy.str2et("2033-11-29"), 600.0
    )
    occultation = Occultation("JUICE_JANUS", "JUPITER", "SUN")

    vector = np.asarray(occultation(times))
    scalar = np.array([occultation(float(t)) for t in times])

    assert (vector == scalar).all()
