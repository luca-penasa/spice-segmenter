from spice_segmenter import SpiceWindow, log_enable
from spice_segmenter.constraint import ConstraintBase
from spice_segmenter.constraint_solver import GenericScalarSolver
from spice_segmenter.constraint_solver.constraint_solver import SpiceEventSolver, SpiceOccultationSolver, SpiceWindowSolver
from spice_segmenter.trajectory_properties import AngularSize, Distance
from spice_segmenter.coordinates import Vector
from spice_segmenter.constraint_solver import get_appropriate_solver
from spice_segmenter import Occultation, OccultationTypes
from planetary_coverage.spice import SpiceBody

log_enable('DEBUG')

c_less = AngularSize("JUICE_JANUS", "CALLISTO").as_unit("deg") < 2
c_less_def = AngularSize("JUICE_JANUS", "CALLISTO").as_unit("deg") < '2 deg'
c_gt = AngularSize("JUICE_JANUS", "CALLISTO").as_unit("deg") > 2

c_d = Distance('JUICE_JANUS', 'CALLISTO') < '5000 km'

p_lat = Vector('CALLISTO', 'JUICE_JANUS', frame=SpiceBody('CALLISTO').frame).as_latitudinal.latitude

c_lat = p_lat < '20 deg'

w = SpiceWindow.from_start_end("2032-01-01T00:00:00", "2033-01-01T00:00:00")

p_occ = Occultation('JUICE_JANUS', 'CALLISTO', 'JUPITER')

c_occ = p_occ == OccultationTypes.ANNULAR

from spice_segmenter import config

config.solver_step = 48 * 60 * 60  # 48 hours to make it faster


def _test_solve_with_generic_scalar_solver(constraint: ConstraintBase) -> None:
    solver = GenericScalarSolver(constraint=constraint)
    got = solver.solve(w)

    astab = got.to_pandas()
    print("Found intervals:")
    print(astab)

    assert len(astab) >= 1


def test_generic_scalar_solver() -> None:
    _test_solve_with_generic_scalar_solver(c_less)
    _test_solve_with_generic_scalar_solver(c_gt)


def test_solver_selection() -> None:
    # an angular size constraint should go to the generic scalar solver, for now.
    # this will be improved when a simplification of the constraint to the ones directly
    # supported by spice event solver will be implemented.
    # meaning this will fail if we ever do so
    solver =get_appropriate_solver(c_less)
    assert (solver == GenericScalarSolver)

    # solving a distance should use SpiceEventSolver
    solver = get_appropriate_solver(c_d)
    assert(solver == SpiceEventSolver)

    # also coordinates should be solved via the spice event solver
    solver = get_appropriate_solver(c_lat)
    assert(solver == SpiceEventSolver)

    # a combination of constraints will just solve their spice windows
    c_comb = c_d & c_lat
    solver = get_appropriate_solver(c_comb)
    assert(solver == SpiceWindowSolver)

    # a spice occultation constraint
    solver = get_appropriate_solver(c_occ)
    assert(solver == SpiceOccultationSolver)


