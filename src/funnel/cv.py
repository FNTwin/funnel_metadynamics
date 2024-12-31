from abc import ABC, abstractmethod
from typing import List, Optional

from openmm.app.metadynamics import BiasVariable
import openmm

class CVInterface(ABC):
    """
    Abstract base class for implementing bias variables in molecular dynamics simulations.
    
    This class provides the foundation for creating custom bias variables used in 
    enhanced sampling methods like metadynamics. It handles the creation and management
    of collective variables (CVs) and their associated restraint forces.

    Attributes:
        n_components: Number of components in the bias variable
        restraint_forces: List of OpenMM forces used for restraints
        cv: List of bias variables used in the simulation
    """
    n_components: int
    restraint_forces: List[openmm.Force] = []
    cv: List[Optional[BiasVariable]] = []

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        """Initialize the CV list and construct bias forces after instance creation."""
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError(f"n_components must be a positive integer, got {self.n_components}")
        
        self.cv = [None] * self.n_components
        self.construct_bias_forces()

    def get_cv(self) -> List[BiasVariable]:
        """
        Retrieve the list of collective variables.
        
        Returns:
            List[BiasVariable]: List of defined collective variables
            
        Raises:
            ValueError: If any CV component is undefined
        """
        self._validate_cv_components()
        return self.cv

    def get_cv_bias(self) -> List[float]:
        """
        Get the bias width for each collective variable.
        
        Returns:
            List[float]: List of bias widths for each CV
        """
        return [cv.biasWidth for cv in self.get_cv()]

    def get_restraint(self) -> List[openmm.Force]:
        """
        Retrieve the restraint forces.
        
        Returns:
            List[openmm.Force]: List of OpenMM forces used for restraints
        """
        return self.restraint_forces

    def construct_bias_forces(self) -> None:
        """
        Construct both restraint forces and collective variables.
        This method calls make_restraint() and make_cv() in sequence.
        """
        self.make_restraint()
        self.make_cv()

    def _validate_cv_components(self) -> None:
        """
        Validate that all CV components are properly defined.
        
        Raises:
            ValueError: If the number of CVs doesn't match n_components or if any CV is undefined
        """
        if len(self.cv) != self.n_components:
            raise ValueError(
                f"Number of CV components mismatch: expected {self.n_components}, got {len(self.cv)}"
            )
            
        undefined_cvs = [i for i, cv in enumerate(self.cv) if cv is None]
        if undefined_cvs:
            raise ValueError(
                f"Undefined CV components at indices: {undefined_cvs}"
            )

    def get_correction_term(self) -> float:
        """
        Calculate correction term for the bias potential.
        Override this method in child classes if correction is needed.
        
        Returns:
            float: Correction term value (default: 0.0)
        """
        return 0.0

    @abstractmethod
    def make_cv(self) -> None:
        """
        Create the collective variables.
        Must be implemented by child classes.
        """
        raise NotImplementedError("make_cv() must be implemented by child class")

    @abstractmethod
    def make_restraint(self) -> None:
        """
        Create the restraint forces.
        Must be implemented by child classes.
        """
        raise NotImplementedError("make_restraint() must be implemented by child class")