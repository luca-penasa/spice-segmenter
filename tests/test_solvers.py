from spice_segmenter import SpiceWindow
from spice_segmenter.constraint_solver import ConstraintBase, GenericScalarSolver
from spice_segmenter.trajectory_properties import AngularSize

c_less = AngularSize("JUICE_JANUS", "CALLISTO").as_unit("deg") < 2
c_gt = AngularSize("JUICE_JANUS", "CALLISTO").as_unit("deg") > 2


w = SpiceWindow.from_start_end("2032-01-01T00:00:00", "2033-01-01T00:00:00")


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
