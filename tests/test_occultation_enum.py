from spice_segmenter.occultation import OccultationTypes


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
