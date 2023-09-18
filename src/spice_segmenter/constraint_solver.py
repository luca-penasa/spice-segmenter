from typing import Iterable, Optional, Protocol, Type

import pint
import spiceypy
from attr import define, field
from loguru import logger as log
from planetary_coverage import SpiceRef
from spiceypy import gfrefn, gfstep

from spice_segmenter.ops import Inverted

from .occultation import OccultationTypes
from .search_reporter import SearchReporter, get_default_reporter_class
from .spice_window import SpiceWindow
from .trajectory_properties import Constant, Constraint, ConstraintBase, ConstraintTypes


class Solver(Protocol):
    def __init__(self, constraint: ConstraintBase, step: float) -> None:
        ...

    def solve(self, window: SpiceWindow) -> SpiceWindow:
        ...

    @staticmethod
    def can_solve(constraint: ConstraintBase) -> bool:
        ...


@define(repr=False, order=False, eq=False)
class GfevntSolverConfigurator:
    names: list[str] = field(factory=list)
    strings: list[str] = field(factory=list)
    floats: list[float] = field(factory=list)
    integers: list[int] = field(factory=list)
    booleans: list[bool] = field(factory=list)
    operator: str = ""
    refval: float = 0.0
    quantity: str = ""
    adjust: float = 0.0

    def add_str_parameter(self, name: str, value: str) -> None:
        log.debug("adding str parameter {} with value {}", name, value)
        if name in self.names:
            raise ValueError(f"Parameter {name} already exists")

        self.names.append(name)
        self.strings.append(value)

    def add_vector_parameter(self, name: str, vector: Iterable[float]) -> None:
        log.debug("adding vector parameter {} with value {}", name, vector)
        if name in self.names:
            raise ValueError(f"Parameter {name} already exists")

        self.names.append(name)
        for value in vector:
            self.floats.append(value)

    def as_dict(self) -> dict:
        return {
            "qpnams": self.names,
            "qcpars": self.strings,
            "qdpars": self.floats,
            "qipars": self.integers,
            "qlpars": self.booleans,
            "qnpars": len(self.names),
            "op": self.operator,
            "refval": self.refval,
            "gquant": self.quantity,
            "adjust": self.adjust,
        }

    def set_from_dict(self, pars: dict) -> None:
        log.debug("Setting config from pars {}", pars)

        quantity = pars["property"]

        log.debug("setting parameters for a property {}", quantity)

        if quantity not in self.known_properties():
            raise ValueError("Unknown property {}", quantity)

        if quantity.lower() == "distance":
            self.set_distance_from_dict(pars)

        elif quantity.lower() == "phase_angle":
            self.set_phase_angle_from_dict(pars)

        elif quantity.lower() == "coordinate":
            self.set_coordinate_from_dict(pars)

        else:
            raise ValueError(f"unsupported property {quantity}")

        self.quantity = quantity.replace("_", " ").upper()

        operator = pars["operator"]
        operator = self.translate_minamx(operator)

        is_min_max = True if operator in self.minmax_operators() else False

        log.debug("Operator {} ", operator)
        if operator not in self.known_operators():
            raise ValueError(f"Unknown operator {operator}")

        self.operator = operator

        if self.operator in ["absmin", "absmax"]:
            self.adjust = pars["adjust"]
            log.debug("Using {} as adjust", self.adjust)

        if not is_min_max:
            refval = pars.get("reference_value", None)
            if refval is None:
                raise ValueError(
                    "No reference value found, and not a minmax constraint"
                )

            refval_unit = pars.get("reference_value_unit", None)
            property_unit = pars.get("property_unit", None)

            if refval_unit is None:
                log.error("No reference value unit found")
                raise ValueError("No reference value unit found")

            if property_unit is None:
                log.error("No property unit found")
                raise ValueError("No property unit found")

            log.debug("property unit {}, refval unit {}", property_unit, refval_unit)
            if refval_unit != property_unit and not refval_unit == "dimensionless":
                log.debug(
                    "converting refval with unit {} to property unit {}",
                    refval_unit,
                    property_unit,
                )

                refval = pint.Quantity(refval, refval_unit).to(property_unit).magnitude

            self.refval = float(refval)

    def set_distance_from_dict(self, pars: dict) -> None:
        self.add_str_parameter("TARGET", pars["target"])
        self.add_str_parameter("OBSERVER", pars["observer"])
        self.add_str_parameter("ABCORR", pars["abcorr"])

    def set_phase_angle_from_dict(self, pars: dict) -> None:
        self.add_str_parameter("TARGET", pars["target"])
        self.add_str_parameter("OBSERVER", pars["observer"])
        self.add_str_parameter("ABCORR", pars["abcorr"])
        self.add_str_parameter("ILLUM", pars["third_body"])

    def set_coordinate_from_dict(self, pars: dict) -> None:
        self.add_str_parameter("TARGET", pars["target"])
        self.add_str_parameter("OBSERVER", pars["origin"])
        self.add_str_parameter("ABCORR", pars["abcorr"])
        self.add_str_parameter("COORDINATE SYSTEM", pars["coordinate_type"])
        self.add_str_parameter(
            "COORDINATE", pars["component"].upper().replace("_", " ")
        )
        self.add_str_parameter("REFERENCE FRAME", pars["frame"])
        self.add_str_parameter("VECTOR DEFINITION", pars["vector_definition"])
        self.add_str_parameter("METHOD", pars["method"])

        if pars["vector_definition"] == "SURFACE INTERCEPT POINT".lower().replace(
            " ", "_"
        ):
            log.error("Surface intercept point not implemented")
            raise NotImplementedError

        else:
            self.add_str_parameter("DREF", "")
            self.add_vector_parameter("DVEC", [0.0, 0, 0])
        # "DREF",
        # "DVEC"

    @staticmethod
    def known_properties() -> list[str]:
        return [
            "angular_separation",
            "coordinate",
            "distance",
            "illumination_angle",
            "phase_angle",
            "range_rate",
        ]

    @staticmethod
    def known_operators() -> list[str]:
        return [">", "=", "<", "absmax", "absmin", "locmax", "locmin"]

    @staticmethod
    def minmax_operators() -> list[str]:
        return ["absmax", "absmin", "locmax", "locmin"]

    @staticmethod
    def translate_minamx(value: str) -> str:
        conf_mapping = {
            "local_minimum": "locmin",
            "local_maximum": "locmax",
            "global_minimum": "absmin",
            "global_maximum": "absmax",
        }

        try:
            return conf_mapping[value]

        except KeyError:
            return value


@define(repr=False, order=False, eq=False)
class GfevntSolver:
    constraint: Optional[ConstraintBase]
    step: float = 60 * 60  # in seconds
    config: dict = field(factory=dict)
    result: SpiceWindow | None = None
    reporter: SearchReporter = field(factory=get_default_reporter_class)

    def solve(self, window: SpiceWindow) -> SpiceWindow:
        log.debug(
            "Solvig with gfevnt {} and type {}", self.constraint, type(self.constraint)
        )
        if not self.constraint:
            log.error("You need to provide a valid constraint")
            raise ValueError

        cfg: dict = {}
        self.constraint.config(cfg)
        log.debug("Config {}", cfg)

        constraint_config = self.configure()

        log.debug("Solver config {}", self.config)

        maxval = 100000
        cnfine = window.spice_window  # the window
        self.result = SpiceWindow(size=maxval)  # the resulting window

        spiceypy.gfsstp(self.step)  # set the step size

        spiceypy.gfevnt(
            udstep=spiceypy.utils.callbacks.SpiceUDFUNS(gfstep),
            udrefn=spiceypy.utils.callbacks.SpiceUDREFN(gfrefn),
            lenvals=100,
            tol=1e-3,
            rpt=True,
            udrepi=self.reporter.init_search_spice,
            udrepu=self.reporter.update_function_spice,
            udrepf=self.reporter.end_search_spice,
            nintvls=maxval,
            bail=True,
            udbail=spiceypy.utils.callbacks.SpiceUDBAIL(spiceypy.gfbail),
            cnfine=cnfine,
            result=self.result.spice_window,
            **self.config,
        )

        inverted = constraint_config.get("inverted", False)
        if inverted:
            log.debug("INVERTING RESULT")
            self.result = self.result.complement(window)

        return self.result

    def configure(self) -> dict:
        log.debug("Configuring solver")

        config: dict = {}
        if not self.constraint:
            log.error("You need to provide a valid constraint")
            raise ValueError

        self.constraint.config(config)

        log.debug("Constraint full config {}", config)

        pars_composer = GfevntSolverConfigurator()
        pars_composer.set_from_dict(config)
        self.config = pars_composer.as_dict()

        log.debug("config: \n {}", self.config)

        return config

    @staticmethod
    def can_solve(constraint: ConstraintBase) -> bool:
        if constraint.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            return False

        conf: dict = {}
        constraint.config(conf)

        quantity = conf.get("property", None)
        if not quantity:
            log.debug("No property keyword found in constraint, this might imply a bug")
            return False

        if quantity.lower() in GfevntSolverConfigurator.known_properties():
            return True

        return False


@define(repr=False, order=False, eq=False)
class GfocceSolver:
    """Occultation Solver"""

    constraint: Constraint | None = None
    step: float = 60 * 60  # in seconds
    config: dict = field(factory=dict)
    result: SpiceWindow | None = None
    reporter: SearchReporter = field(factory=get_default_reporter_class)

    def configure(self) -> dict:
        """
        occtyp     I   Type of occultation.
        front      I   Name of body occulting the other.
        fshape     I   Type of shape model used for front body.
        fframe     I   Body-fixed, body-centered frame for front body.
        back       I   Name of body occulted by the other.
        bshape     I   Type of shape model used for back body.
        bframe     I   Body-fixed, body-centered frame for back body.
        abcorr     I   Aberration correction flag.
        obsrvr     I   Name of the observing body.
        tol        I   Convergence tolerance in seconds.
        udstep     I   Name of the routine that returns a time step.
        udrefn     I   Name of the routine that computes a refined time.
        rpt        I   Progress report flag.
        udrepi     I   Function that initializes progress reporting.
        udrepu     I   Function that updates the progress report.
        udrepf     I   Function that finalizes progress reporting.
        bail       I   Logical indicating program interrupt monitoring.
        udbail     I   Name of a routine that signals a program interrupt.
        cnfine    I-O  SPICE window to which the search is restricted.
        result     O   SPICE window containing results.
        """

        if not self.constraint:
            log.error("You need to provide a valid constraint")
            raise ValueError

        config: dict = {}
        self.constraint.config(config)  # extract the config

        self.constraint.left

        self.config.update(
            {
                "front": config["front"],
                "fshape": "ELLIPSOID",
                "fframe": SpiceRef(config["front"]).frame,
                "back": config["back"],
                "bshape": "ELLIPSOID",
                "bframe": SpiceRef(config["back"]).frame,
                "abcorr": config["light_time_correction"],
                "obsrvr": config["observer"],
                "tol": 1e-3,
                "udstep": spiceypy.utils.callbacks.SpiceUDFUNS(gfstep),
                "udrefn": spiceypy.utils.callbacks.SpiceUDREFN(gfrefn),
                "rpt": True,
                "udrepi": self.reporter.init_search_spice,
                "udrepu": self.reporter.update_function_spice,
                "udrepf": self.reporter.end_search_spice,
                "bail": True,
                "udbail": spiceypy.utils.callbacks.SpiceUDBAIL(spiceypy.gfbail),
            }
        )

        log.debug("Configured: {}", self.config)

        return config

    def solve(self, window: SpiceWindow) -> SpiceWindow:
        if not self.constraint or not self.can_solve(self.constraint):
            log.error("You need to provide a valid constraint/constraints not solvable")
            raise ValueError

        constraint_config = self.configure()

        maxval = 10000
        cnfine = window.spice_window  # the window
        self.result = SpiceWindow(size=maxval)  # the resulting window

        spiceypy.gfsstp(self.step)  # set the step size

        right = self.constraint.right

        if not isinstance(right, Constant):
            raise TypeError

        if not isinstance(right.value, OccultationTypes):
            raise TypeError

        right_value = right.value

        if right_value.value == OccultationTypes.ANY.value:
            occtyp = "ANY"
        elif right_value.value == OccultationTypes.FULL.value:
            occtyp = "FULL"
        elif right_value.value == OccultationTypes.PARTIAL.value:
            occtyp = "PARTIAL"
        elif right_value.value == OccultationTypes.ANNULAR.value:
            occtyp = "ANNULAR"
        else:
            log.debug("Unknown occultation type {}", right_value)
            raise ValueError

        self.config["occtyp"] = occtyp
        self.config["cnfine"] = cnfine
        self.config["result"] = self.result.spice_window

        log.debug("Configured occultation solver: {}", self.config)

        spiceypy.gfocce(**self.config)

        inverted = constraint_config.get("inverted", False)
        if inverted:
            log.debug("INVERTING RESULT")
            self.result = self.result.complement(window)

        return self.result

    @staticmethod
    def can_solve(constraint: ConstraintBase) -> bool:
        if constraint.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            return False

        if constraint.left.name in ["occultation"]:
            log.debug("can solve {}", constraint.left.name)
            return True

        return False


@define(repr=False, order=False, eq=False)
class SpiceWindowSolver:
    """Solves a constraints made by two constraints returing SpiceWindow objects"""

    constraint: ConstraintBase | None = None
    step: float = 60 * 60  # in seconds

    def solve(self, window: SpiceWindow) -> SpiceWindow:
        if not self.constraint or not self.can_solve(self.constraint):
            log.error("No constraint set or constraint cannot be solved")
            raise ValueError

        if not isinstance(self.constraint.left, ConstraintBase) or not isinstance(
            self.constraint.right, ConstraintBase
        ):
            # a double check to make sure we are dealing with constraints that can be processed
            log.error("Left or right operands not of type Constraint")
            raise TypeError

        solver: Type[Solver] = get_appropriate_solver(self.constraint.left)

        # solve the first constraint
        result = solver(self.constraint.left, step=self.step).solve(window)

        assert result is not None

        solver = get_appropriate_solver(self.constraint.right)
        if self.constraint.operator == "&":
            log.debug("solving an AND operator")
            result2 = solver(self.constraint.right, step=self.step).solve(result)
            op_res = result.intersect(result2)

        elif self.constraint.operator == "|":
            log.debug("solving an OR operator")
            result2 = solver(self.constraint.right, step=self.step).solve(
                result.complement(window)
            )
            op_res = result.union(result2)

        else:
            log.error("Operator {} not implemented", self.constraint.operator)
            raise NotImplementedError

        if isinstance(self.constraint, Inverted):
            log.debug("INVERTING RESULT")
            op_res = op_res.complement(window)  # invert the result

        return op_res

    @staticmethod
    def can_solve(constraint: ConstraintBase) -> bool:
        if constraint.ctype == ConstraintTypes.COMPARE_TO_OTHER_CONSTRAINT:
            log.debug("Found constraint to constraint comparison")
            return True
        return False


@define(repr=False, order=False, eq=False)
class MasterSolver:
    constraint: ConstraintBase | None = None
    step: float = 60 * 60
    minimum_interval_size: float = 0.0  # seconds

    def solve(self, window: SpiceWindow) -> SpiceWindow:
        if not self.constraint or not self.can_solve(self.constraint):
            log.error("No constraint set or constraint not solvable")
            raise ValueError

        solver = get_appropriate_solver(self.constraint)
        result = solver(self.constraint, step=self.step).solve(window)

        if self.minimum_interval_size > 0.0:
            log.debug("Removing small intervals")
            result.remove_small_intervals(self.minimum_interval_size)

        return result

    def can_solve(self, constraint: ConstraintBase) -> bool:
        if get_appropriate_solver(constraint):
            return True
        return False


solvers: list[Type[Solver]] = [GfevntSolver, GfocceSolver, SpiceWindowSolver]


def get_appropriate_solver(constraint: ConstraintBase) -> Type[Solver]:
    for solver in solvers:
        if solver.can_solve(constraint):
            log.debug(
                "Found solver {} for constraint {}, of type {}",
                solver,
                constraint,
                type(constraint),
            )
            return solver

    raise NotImplementedError(f"No solver implemented for constraint {constraint}")
