from typing import List

from dataclasses import dataclass, field
import openmm
import numpy as np
from abc import abstractmethod
from openmm.app.metadynamics import BiasVariable

from funnel.cv import CVInterface

from openmm.unit import nanometers


@dataclass
class FunnelPotentialInterface(CVInterface):
    """
    Base class for implementing funnel-based bias potentials in molecular dynamics simulations.
    
    The funnel potential is used to enhance sampling in specific regions of interest by creating
    a funnel-shaped bias along a chosen path. 
    
    Attributes:
        p1: List of atom indices defining the starting point of the funnel
        p2: List of atom indices defining the end point of the funnel
        lig: List of atom indices defining the ligand
        lower_wall: Lower boundary of the funnel (in nm)
        upper_wall: Upper boundary of the funnel (in nm)
        wall_width: Width of the funnel walls (in nm)
        wall_buffer: Buffer distance between walls and funnel (in nm)
        alpha: Sigmoid inflection parameter
        s: Distance from P1 to sigmoid inflection point (in nm)
        k1: Force constant for lower/upper wall restraints (kJ/mol/nm²)
        k2: Force constant for radial restraints (kJ/mol/nm²)
        hill_width: Width of metadynamics Gaussian hills (in nm)
        sigma_width: Width of radial Gaussian sigma (in nm)
        grid_width: Number of grid points for Gaussian hills
    """
    p1: List[float]
    p2: List[float]
    lig: List[int]
    lower_wall: float
    upper_wall: float
    wall_width: float
    wall_buffer: float
    alpha: float
    s: float
    k1: float = 10000.0
    k2: float = 1000.0
    hill_width: float = 0.025
    sigma_width: float = 0.05
    grid_width: int = 200
    cv_forces: List[openmm.Force] = field(default_factory= lambda : [])

    def __post_init__(self):
        """Initialize funnel parameters and construct forces."""
        self._validate_inputs()
        self.make_funnel()
        super().__init__(n_components=2)

    def _validate_inputs(self) -> None:
        """Validate input parameters for the funnel potential."""
        if not all(isinstance(x, (list, np.ndarray)) for x in [self.p1, self.p2, self.lig]):
            raise ValueError("p1, p2, and lig must be lists or numpy arrays")
        
        if self.lower_wall >= self.upper_wall:
            raise ValueError(f"lower_wall ({self.lower_wall}) must be less than upper_wall ({self.upper_wall})")
        
        if any(x < 0 for x in [self.wall_width, self.wall_buffer, self.k1, self.k2]):
            raise ValueError("wall_width, wall_buffer, k1, and k2 must be positive")
        
        if self.grid_width <= 0:
            raise ValueError(f"grid_width must be more than 0, got {self.grid_width}")

    @abstractmethod
    def make_funnel(self) -> None:
        """Create the funnel forces. Must be implemented by child classes."""
        raise NotImplementedError

    def make_cv(self) -> None:
        """Create collective variables for the funnel."""
        if not self.cv_forces:
            raise ValueError("cv_forces must be created before calling make_cv()")
            
        self.cv[0] = BiasVariable(
            self.cv_forces[0],
            self.lower_wall,
            self.upper_wall,
            self.hill_width,
            False,
            gridWidth=self.grid_width
        )
        
        self.cv[1] = BiasVariable(
            self.cv_forces[1],
            0.0,
            self.wall_width + self.wall_buffer + 0.1,
            self.sigma_width,
            False,
            gridWidth=self.grid_width
        )

    def make_restraint(self) -> None:
        """Create restraint forces for the funnel walls."""
        self.restraint_forces.extend([
            self._create_upper_wall_restraint(),
            self._create_distance_restraint(),
            self._create_lower_wall_restraint()
        ])

    @abstractmethod
    def _create_upper_wall_restraint(self) -> openmm.Force:
        """Create the upper wall restraint force."""
        raise NotImplementedError

    @abstractmethod
    def _create_distance_restraint(self) -> openmm.Force:
        """Create the radial distance restraint force."""
        raise NotImplementedError

    @abstractmethod
    def _create_lower_wall_restraint(self) -> openmm.Force:
        """Create the lower wall restraint force."""
        raise NotImplementedError

    def get_correction_term(self, delta: float = 0.01) -> float:
        """
        Calculate the correction term for free energy calculations in kJ/mol.
        
        Args:
            delta: Spacing between grid points along the projection (in nm)
            
        Returns:
            float: Correction term in kcal/mol
        """
        funnel_cv = np.arange(self.lower_wall, self.upper_wall, delta)
        funnel = self.funnel_f(funnel_cv)
        
        # Calculate volume using trapezoidal integration
        volume = np.trapz(funnel, dx=delta) * np.pi
        
        # Convert to average area in Angstrom
        area = (volume * 1000) / (abs(self.upper_wall - self.lower_wall) * 10)
        
        # Convert to kJ/mol (kT to kcal/mol factor = 0.5922)
        return np.log(area / 1660) * 0.5922

    def funnel_f(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the funnel shape at given positions.
        
        Args:
            x: Array of positions along the funnel axis
            
        Returns:
            np.ndarray: Funnel radius at each position
        """
        return (self.wall_width / (1 + np.exp(self.alpha * (x - self.s)))) + self.wall_buffer

    @property
    def projection(self) -> BiasVariable:
        """Get the projection component of the collective variable."""
        return self.cv[0]

    @property
    def extent(self) -> BiasVariable:
        """Get the extent (radial) component of the collective variable."""
        return self.cv[1]



@dataclass
class StaticFunnelPotential(FunnelPotentialInterface):
    """
    Static funnel potential implementation that uses fixed coordinates for funnel endpoints.
    The funnel is defined by static coordinates rather than atom groups.
    This implementation is based on the methods described in:
        - https://www.pnas.org/doi/10.1073/pnas.1303186110
    """


    def _create_upper_wall_restraint(self) -> openmm.Force:
        """Create the upper wall restraint force."""
        upper_wall_rest = openmm.CustomCentroidBondForce(
            1,
            "(k/2)*max(pos - upper_wall, 0)^2;"
            "pos = distance(g1,lower)*cos(angle_0);"
            "angle_0 = angle(g1,lower,upper);"
            "lower = lower_x, lower_y, lower_z;"
            "upper = upper_x, upper_y, upper_z;",
        )
        lig_idx = upper_wall_rest.addGroup(self.lig)

        for i, (coord, prefix) in enumerate(zip([self.p1, self.p2], ['lower', 'upper'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                upper_wall_rest.addGlobalParameter(f"{prefix}_{axis}", coord[j] * nanometers)

        upper_wall_rest.addGlobalParameter("k", self.k1)
        upper_wall_rest.addGlobalParameter("upper_wall", self.upper_wall)
        upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
        return upper_wall_rest



    def _create_distance_restraint(self) -> openmm.Force:
        """Create the radial distance restraint force."""
        dist_restraint = openmm.CustomCentroidBondForce(
            1,
            "(k/2)*max(pos, 0)^2;"
            "pos = distance(g1, lower) * pos_sin - (a / (1 + exponential) + d);"
            "exponential = exp(b * (distance(g1, lower) * pos_cos - c));"
            "pos_sin = sin(angle_0);"
            "pos_cos = cos(angle_0);"
            "angle_0 = angle(g1,lower,upper);"
            "lower = lower_x, lower_y, lower_z;"
            "upper = upper_x, upper_y, upper_z;",
        )
        lig_idx = dist_restraint.addGroup(self.lig)

        for i, (coord, prefix) in enumerate(zip([self.p1, self.p2], ['lower', 'upper'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                dist_restraint.addGlobalParameter(f"{prefix}_{axis}", coord[j] * nanometers)

        dist_restraint.addGlobalParameter("k", self.k2)
        dist_restraint.addGlobalParameter("a", self.wall_width)  # from cv
        dist_restraint.addGlobalParameter("b", self.alpha)  # from cv
        dist_restraint.addGlobalParameter("c", self.s)  # from cv
        dist_restraint.addGlobalParameter("d", self.wall_buffer)  # from cv
        dist_restraint.setUsesPeriodicBoundaryConditions(True)
        return dist_restraint

    def _create_lower_wall_restraint(self) -> openmm.Force:
        """Create the lower wall restraint force."""
        lower_wall_rest = openmm.CustomCentroidBondForce(
            1,
            "(k/2)*min(pos - lower_wall, 0)^2;"
            "pos = distance(g1,lower)*cos(angle_0);"
            "angle_0 = angle(g1,lower,upper);"
            "lower = lower_x, lower_y, lower_z;"
            "upper = upper_x, upper_y, upper_z;",
        )
        lig_idx = lower_wall_rest.addGroup(self.lig)

        for i, (coord, prefix) in enumerate(zip([self.p1, self.p2], ['lower', 'upper'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                lower_wall_rest.addGlobalParameter(f"{prefix}_{axis}", coord[j] * nanometers)

        lower_wall_rest.addGlobalParameter("k", self.k1)
        lower_wall_rest.addGlobalParameter("lower_wall", self.lower_wall)
        lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
        return lower_wall_rest

    
    def make_funnel(self) -> None:
        """Create funnel forces using static coordinates."""
        # Projection force
        funnel = openmm.CustomCentroidBondForce(
            1,
            "distance(g1,lower)*cos(angle_0);"
            "angle_0 = angle(g1,lower,upper);"
            "lower = lower_x, lower_y, lower_z;"
            "upper = upper_x, upper_y, upper_z;",
        )
        funnel.addGroup(self.lig)
        
        # Add coordinates as global parameters
        for i, (coord, prefix) in enumerate(zip([self.p1, self.p2], ['lower', 'upper'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                funnel.addGlobalParameter(f"{prefix}_{axis}", coord[j] * nanometers)
        
        funnel.setUsesPeriodicBoundaryConditions(True)
        
        # Extent force
        reaction = openmm.CustomCentroidBondForce(
            1,
            "distance(g1,lower)*sin(angle_0);"
            "angle_0 = angle(g1,lower,upper);"
            "lower = lower_x, lower_y, lower_z;"
            "upper = upper_x, upper_y, upper_z;",
        )
        reaction.addGroup(self.lig)
        
        # Add coordinates as global parameters
        for i, (coord, prefix) in enumerate(zip([self.p1, self.p2], ['lower', 'upper'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                reaction.addGlobalParameter(f"{prefix}_{axis}", coord[j] * nanometers)
        
        reaction.setUsesPeriodicBoundaryConditions(True)
        
        self.cv_forces.extend([funnel, reaction])


@dataclass
class SigmoidFunnelPotential(FunnelPotentialInterface):
    """
    Sigmoid funnel potential implementation that uses atom groups for funnel endpoints.
    The funnel adapts to the movement of the specified atom groups.
    This implementation is based on the methods described in:
        - REF1: https://pubs.acs.org/doi/10.1021/acs.jcim.6b00772
        - REF2: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7467642/
    """
    
    def make_funnel(self) -> None:
        """Create funnel forces using dynamic atom groups."""
        # Projection force
        funnel = openmm.CustomCentroidBondForce(
            3,
            "distance(g1,g2)*cos(angle_0);"
            "angle_0 = angle(g1,g2,g3);"
        )
        funnel.addGroup(self.lig)
        funnel.addGroup(self.p1)
        funnel.addGroup(self.p2)
        funnel.addBond([0, 1, 2])  # Connect groups in order: lig, p1, p2
        funnel.setUsesPeriodicBoundaryConditions(True)
        
        # Extent force
        reaction = openmm.CustomCentroidBondForce(
            3,
            "distance(g1,g2)*sin(angle_0);"
            "angle_0 = angle(g1,g2,g3);"
        )
        reaction.addGroup(self.lig)
        reaction.addGroup(self.p1)
        reaction.addGroup(self.p2)
        reaction.addBond([0, 1, 2])  # Connect groups in order: lig, p1, p2
        reaction.setUsesPeriodicBoundaryConditions(True)
        
        self.cv_forces.extend([funnel, reaction])


    def _create_upper_wall_restraint(self) -> openmm.Force:
        """Create the upper wall restraint force."""
        upper_wall_rest = openmm.CustomCentroidBondForce(
            3, "(k/2)*max(pos - upper_wall, 0)^2;" "pos = distance(g1,g2)*cos(angle(g1,g2,g3));"
        )
        lig_idx = upper_wall_rest.addGroup(self.lig)
        p1_idx = upper_wall_rest.addGroup(self.p1)
        p2_idx = upper_wall_rest.addGroup(self.p2)
        upper_wall_rest.addBond([lig_idx, p1_idx, p2_idx])
        upper_wall_rest.addGlobalParameter("k", self.k1)
        upper_wall_rest.addGlobalParameter("upper_wall", self.upper_wall)
        upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
        return upper_wall_rest


    def _create_distance_restraint(self) -> openmm.Force:
        """Create the radial distance restraint force."""
        dist_restraint = openmm.CustomCentroidBondForce(
            3,
            "(k/2)*max(pos, 0)^2;"
            "pos = distance(g1, g2) * pos_sin - (a / (1 + exponential) + d);"
            "exponential = exp(b * (distance(g1, g2) * pos_cos - c));"
            "pos_sin = sin(angle(g1, g2, g3));"
            "pos_cos = cos(angle(g1, g2, g3));",
        )
        lig_idx = dist_restraint.addGroup(self.lig)
        p1_idx = dist_restraint.addGroup(self.p1)
        p2_idx = dist_restraint.addGroup(self.p2)
        dist_restraint.addBond([lig_idx, p1_idx, p2_idx])
        dist_restraint.addGlobalParameter("k", self.k2)
        dist_restraint.addGlobalParameter("a", self.wall_width)  # from cv
        dist_restraint.addGlobalParameter("b", self.alpha)  # from cv
        dist_restraint.addGlobalParameter("c", self.s)  # from cv
        dist_restraint.addGlobalParameter("d", self.wall_buffer)  # from cv
        dist_restraint.setUsesPeriodicBoundaryConditions(True)
        return dist_restraint



    def _create_lower_wall_restraint(self) -> openmm.Force:
        """Create the lower wall restraint force."""
        lower_wall_rest = openmm.CustomCentroidBondForce(
            3, "(k/2)*min(pos - lower_wall, 0)^2;" "pos = distance(g1,g2)*cos(angle(g1,g2,g3));"
        )
        lig_idx = lower_wall_rest.addGroup(self.lig)
        p1_idx = lower_wall_rest.addGroup(self.p1)
        p2_idx = lower_wall_rest.addGroup(self.p2)
        lower_wall_rest.addBond([lig_idx, p1_idx, p2_idx])
        lower_wall_rest.addGlobalParameter("k", self.k1)
        lower_wall_rest.addGlobalParameter("lower_wall", self.lower_wall)
        lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
        return lower_wall_rest
    

