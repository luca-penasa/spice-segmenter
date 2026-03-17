from spice_segmenter import Distance, TimeSegmentsCollection
from spice_segmenter.properties.ring_properties import RingAnsaePhaseGreaterThan
from spice_segmenter.support.config import config
from . import tour_config as tc

config.solver_step = '24 h'
start, end = tc.coverage
tc.load_kernels()

w = TimeSegmentsCollection.from_start_end('2032-01-01', '2034-01-01')


def test_with_unit_or_not():
    d = Distance("JUICE_JANUS", "JUPITER")

    c1 = d < 1000000
    got1 = c1.solve(w)
    N1 = len(got1)


    c2 = d < '1000000 km'
    got2 = c2.solve(w)
    N2 = len(got2)

    c3 = d < '1000000000 m'
    got3 = c3.solve(w)
    N3 = len(got3)

    c4 = d < '100000000000 cm'
    got4 = c4.solve(w)
    N4 = len(got4)

    assert(N1 == N2)
    assert(N1 == N3)
    assert(N1 == N4)
    
def test_1():
    c = RingAnsaePhaseGreaterThan(170) == True