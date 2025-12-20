
from functools import cached_property

from attrs import define, field

from ..properties.observation_properties import (
    AngularSize,
    Distance,
    PhaseAngle,
    SubObserverIlluminationAngles,
    SubObserverPixelScale,
    SubObserverPointVelocity,
)
from ..properties.occultation_types import Occultation, OccultationTypes
from ..properties.reflector_properties import JupiterShineIdealCondition


@define
class TargetProperties:
    target: str = field(converter=lambda x: str(x).upper())
    observer: str = field(default="JUICE_JANUS", converter=lambda x: str(x).upper())

    @cached_property
    def distance(self) -> Distance:
        return Distance(self.observer, self.target)

    @cached_property
    def phase_angle(self) -> PhaseAngle:
        return PhaseAngle(self.observer, self.target)

    @cached_property
    def angular_size(self) -> AngularSize:
        return AngularSize(self.observer, self.target)

    @cached_property
    def pixel_scale(self) -> SubObserverPixelScale:
        return SubObserverPixelScale(observer=self.observer, target=self.target)

    @cached_property
    def sub_sc_velocity(self) -> SubObserverPointVelocity:
        return SubObserverPointVelocity(self.observer, self.target)

    @cached_property
    def sub_sc_illumination_angles(self) -> SubObserverIlluminationAngles:
        return SubObserverIlluminationAngles(self.observer, self.target)

    @cached_property
    def jupiter_shine_ideal_condition(self) -> JupiterShineIdealCondition:
        return JupiterShineIdealCondition(self.observer, self.target)


@define
class OccultationProperties:
    """
    Collection of occultation properties for a target observed from an observer.

    This class provides convenient access to occultation checks for multiple
    occultors as seen from a spacecraft observer.

    Parameters
    ----------
    target : str
        The target body being observed (e.g., "IO", "EUROPA")
    observer : str
        The observing spacecraft (default: "JUICE")
    occultors : list[str], optional
        List of occulting bodies
        (default: ["JUPITER", "IO", "EUROPA", "GANYMEDE", "CALLISTO"])
    light_time_correction : str
        Light time and stellar aberration correction mode (default: "NONE")
        Options: "NONE", "LT", "LT+S", "CN", "CN+S", "XLT", "XLT+S",
        "XCN", "XCN+S"

    Examples
    --------
    >>> occ = OccultationProperties("IO", "JUICE")
    >>> # Check if IO is fully occulted by Jupiter
    >>> constraint = occ.by("JUPITER") == OccultationTypes.FULL
    >>> # Get occultation by any body
    >>> fully_hidden = occ.fully_occulted_by_any()

    >>> # Custom occultors
    >>> occ = OccultationProperties(
    ...     "AMALTHEA", "JUICE", occultors=["JUPITER", "IO"]
    ... )
    >>> constraint = occ.by("IO") == OccultationTypes.FULL
    """

    target: str = field(converter=lambda x: str(x).upper())
    observer: str = field(default="JUICE", converter=lambda x: str(x).upper())
    occultors: list[str] = field(
        factory=lambda: ["JUPITER", "IO", "EUROPA", "GANYMEDE", "CALLISTO"],
    )
    light_time_correction: str = field(
        default="NONE", converter=lambda x: str(x).upper(),
    )

    def __attrs_post_init__(self):
        """Post-initialization to filter out target from occultors."""
        # Use object.__setattr__ to avoid triggering validators
        filtered_occultors = [
            str(o).upper() for o in self.occultors if str(o).upper() != self.target
        ]
        object.__setattr__(self, "occultors", filtered_occultors)

    def by(self, occultor: str) -> Occultation:
        """
        Get occultation of target by a specific occulting body.

        Parameters
        ----------
        occultor : str
            Name of the occulting body

        Returns
        -------
        Occultation
            Occultation object for the target by the specified occultor
        """
        return Occultation(
            self.observer,
            str(occultor).upper(),
            self.target,
            self.light_time_correction,
        )

    def all_occultations(self) -> list[Occultation]:
        """
        Get occultations by all configured occultors.

        Returns
        -------
        list[Occultation]
            List of Occultation objects for all occultors
        """
        return [self.by(occultor) for occultor in self.occultors]

    def fully_occulted_by_any(self, occultation_type=None):
        """
        Create a constraint for when target is occulted by any configured body.

        Parameters
        ----------
        occultation_type : OccultationTypes, optional
            Type of occultation to check for (default: OccultationTypes.FULL)
            Options: FULL, PARTIAL, ANNULAR, ANY

        Returns
        -------
        Constraint
            Combined constraint checking occultation by any occultors
        """
        if occultation_type is None:
            occultation_type = OccultationTypes.FULL

        if not self.occultors:
            msg = "No occultors configured"
            raise ValueError(msg)

        constraint = self.by(self.occultors[0]) == occultation_type
        for occultor in self.occultors[1:]:
            constraint = constraint | (self.by(occultor) == occultation_type)
        return constraint

    def visible(self):
        """
        Create a constraint for when target is NOT fully occulted by any body.

        Returns
        -------
        Constraint
            Constraint that is True when target is visible (not fully occulted)
        """
        return ~self.fully_occulted_by_any(OccultationTypes.FULL)

