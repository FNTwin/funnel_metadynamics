# Currently Refactoring this GUI code. Please use gui.py for now

from typing import Tuple, List
import numpy as np
import mdtraj as md
import nglview as nv

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from ipywidgets import Layout

from funnel.funnels import SigmoidFunnelPotential

def compute_funnel_surface_mesh(
    self,
    n_angle_samples: int = 30,
    step: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vertices and faces for a triangulated funnel surface mesh.
    
    Args:
        n_angle_samples: Number of points around the circumference
        step: Step size along funnel axis
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: vertices and faces arrays for the mesh
    """
    com1, com2 = self.get_coms()

    # Calculate the vector between p1 and p2
    cv_vec = com2 - com1
    cv_unit_vec = cv_vec / np.linalg.norm(cv_vec)

    # Get orthogonal vectors
    a0, a1 = 1, 1  # Simplified from random for consistent visualization
    vec1 = np.array([a0, a1, -(a0 * cv_vec[0] + a1 * cv_vec[1]) / (cv_vec[2] + 1e-10)])
    unit_a = vec1 / np.linalg.norm(vec1)
    unit_b = np.cross(unit_a, cv_unit_vec)

    # Generate vertices
    vertices = []
    z_steps = np.arange(self.lower_wall.value, self.upper_wall.value + step, step)
    theta_steps = np.linspace(0, 2*np.pi, n_angle_samples)

    for z in z_steps:
        # Calculate radius at this z using sigmoid function
        radius = (self.wall_width.value / (1 + np.exp(self.alpha.value * (z - self.s.value)))) + self.wall_buffer.value
        
        for theta in theta_steps:
            # Calculate parametric point on circle at this z
            point = com1 + cv_unit_vec * z + radius * (np.cos(theta) * unit_a + np.sin(theta) * unit_b)
            vertices.append(point)

    vertices = np.array(vertices)

    # Generate faces (triangles)
    faces = []
    n_points_per_ring = len(theta_steps)
    n_rings = len(z_steps)

    for ring in range(n_rings - 1):
        for point in range(n_points_per_ring):
            # Calculate indices for quad corners
            current = ring * n_points_per_ring + point
            next_point = ring * n_points_per_ring + (point + 1) % n_points_per_ring
            above = (ring + 1) * n_points_per_ring + point
            above_next = (ring + 1) * n_points_per_ring + (point + 1) % n_points_per_ring

            # Create two triangles for the quad
            faces.append([current, next_point, above])
            faces.append([next_point, above_next, above])

    faces = np.array(faces)
    
    return vertices, faces


@dataclass
class FunnelParameters:
    """Container for funnel parameters with validation."""
    p1: List[float] = field(default_factory=list)
    p2: List[float] = field(default_factory=list)
    lig: List[int] = field(default_factory=list)
    alpha: float = 3.0
    s: float = 0.1
    lower_wall: float = 0.0
    upper_wall: float = 3.0
    wall_width: float = 0.5
    wall_buffer: float = 0.1

    def _get_ligand_indices(self, trajectory, ligand_name: Optional[str]) -> List[int]:
        """Get atom indices for the ligand."""
        if ligand_name:
            return trajectory.top.select(f"resname {ligand_name}").tolist()
        
        # Try common ligand names
        for name in ["SLT", "MOL", "UNK"]:
            indices = trajectory.top.select(f"resname {name}").tolist()
            if indices:
                return indices
        return []
    
    @staticmethod
    def compute_funnel_surface(funnel_dict) -> np.array:
        com1, com2 = funnel_dict['p1'], funnel_dict['p2']

        lower_wall = funnel_dict["lower_wall"]
        upper_wall = funnel_dict["upper_wall"]
        wall_width = funnel_dict["wall_width"]
        beta_cent = funnel_dict["alpha"]
        s_cent = funnel_dict["s"]
        wall_buffer = funnel_dict["wall_buffer"]

        n_angle_samples = 30
        step = 0.1

        # Calculate the vector between p1 and p2
        cv_vec = com2 - com1
        cv_unit_vec = cv_vec / np.linalg.norm(cv_vec)

        # Get orthogonal vectors.
        a0 = np.random.randint(1, 10)
        a1 = np.random.randint(1, 10)
        vec1 = np.array([a0, a1, -(a0 * cv_vec[0] + a1 * cv_vec[1]) / cv_vec[2]])
        unit_a = vec1 / np.linalg.norm(vec1)
        unit_b = np.cross(unit_a, cv_unit_vec)

        # Iterate along the vector cv
        funnel_coords = []
        for step in np.arange(lower_wall, upper_wall + step, step):
            # sigmoid function
            radius = (wall_width / (1 + np.exp(beta_cent * (step - s_cent)))) + wall_buffer
            for angle in np.arange(-np.pi, np.pi, 2 * np.pi / n_angle_samples):
                # Calculate parametric functions for this specific case
                coord = com1 + cv_unit_vec * step + radius * (np.cos(angle) * unit_a + np.sin(angle) * unit_b)
                funnel_coords.append(coord)
        return np.array(funnel_coords)

class WidgetFactory:
    """Factory class for creating consistent widgets."""
    
    @staticmethod
    def create_slider(
        description: str,
        value: float,
        min_val: float,
        max_val: float,
        step: float,
        width: str = "auto"
    ) -> widgets.FloatSlider:
        """Create a standardized float slider widget."""
        return widgets.FloatSlider(
            description=description,
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            layout=Layout(height="auto", width=width)
        )
    
    @staticmethod
    def create_button(
        description: str,
        button_style: str,
        width: str = "auto",
        handler: Optional[Callable] = None
    ) -> widgets.Button:
        """Create a standardized button widget."""
        button = widgets.Button(
            description=description,
            button_style=button_style,
            layout=Layout(height="auto", width=width)
        )
        if handler:
            button.on_click(handler)
        return button
    
    @staticmethod
    def create_simple_autobutton(description, button_style, width="auto", b_type=widgets.Button):
        return b_type(
            description=description, button_style=button_style,
              layout=Layout(height="auto", width=width)
        )

class FunnelViewer:
    """Handles the 3D visualization of the funnel."""
    
    def __init__(self, trajectory: md.Trajectory):
        self.trajectory = trajectory
        self.view = nv.NGLWidget(gui=True, height="500px")
        self.surface_enabled = False
        self._setup_initial_view()
        
    def _setup_initial_view(self):
        """Set up the initial visualization state."""
        self.view.add_trajectory(self.trajectory)
        self.view.add_cartoon("protein")
        self.view.add_principal_axes("ligand")
        self.view.center()
        self.view.background = "black"
        self.view.render_image(trim=True, factor=2)
        
    def toggle_surface(self):
        """Toggle protein surface visualization."""
        if not self.surface_enabled:
            self.view.add_surface(selection="protein", opacity=0.4, color="gray")
            self.surface_enabled = True
        else:
            self.view.clear_representations()
            self._setup_initial_view()
            self.surface_enabled = False
            
    def update_funnel_visualization(self, funnel_coords: np.ndarray):
        """Update the funnel visualization with new coordinates."""
        try:
            # Remove existing funnel component if it exists
            self.view.remove_component(self.view.component_1)
        except Exception:
            pass
        
        funnel_traj = self._create_funnel_trajectory(funnel_coords)
        self.view.add_trajectory(funnel_traj)
        
    def add_point_visualization(self, coords: np.ndarray, color: List[float], size: float = 0.4):
        """Add a point visualization at specified coordinates."""
        self.view.shape.add_sphere(coords * 10, color, size)
        
    @staticmethod
    def _create_funnel_trajectory(coords: np.ndarray) -> md.Trajectory:
        """Create a trajectory object from funnel coordinates."""
        import pandas as pd
        df = pd.DataFrame({
            'serial': range(len(coords)),
            'name': [f'HE{i+1}' for i in range(len(coords))],
            'element': 'He',
            'resSeq': 1,
            'resName': 'FNL',
            'chainID': 0
        })
        return md.Trajectory(xyz=coords, topology=md.Topology().from_dataframe(df))

class FunnelDashboard:
    """
    Modern implementation of the funnel metadynamics dashboard.
    Provides an interactive interface for funnel parameter adjustment and visualization.
    """
    
    def __init__(self, trajectory: md.Trajectory, ligand_name: Optional[str] = None):
        """
        Initialize the dashboard.
        
        Args:
            trajectory: MD trajectory to visualize
            ligand_name: Name of the ligand residue (default tries common names)
        """
        self.trajectory = trajectory
        self.params = FunnelParameters()
        self.params.lig = self.params._get_ligand_indices(trajectory, ligand_name)
        self.ligand_indices = self.params._get_ligand_indices(trajectory, ligand_name)
        
        # Initialize components
        self.viewer = FunnelViewer(trajectory)
        self._create_widgets()
        self._setup_layout()
        self._bind_callbacks()
        
    def _create_widgets(self):
        """Create all dashboard widgets."""
        factory = WidgetFactory()
        
        # Parameter sliders
        self.sliders = {
            'lower_wall': factory.create_slider('lower_wall', 0.0, -5.0, 5.0, 0.1),
            'upper_wall': factory.create_slider('upper_wall', 3.0, 0.1, 15.0, 0.1),
            'wall_width': factory.create_slider('wall_width', 0.5, 0.0, 5.0, 0.01),
            'wall_buffer': factory.create_slider('wall_buffer', 0.1, 0.0, 5.0, 0.01),
            'alpha': factory.create_slider('alpha', 3.0, 0.0, 30.0, 0.1, "100%"),
            's': factory.create_slider('s', 0.1, 0.0, 10.0, 0.05, "100%")
        }
        
        # Selection tools
        self.selection_input = widgets.Text(
            description="Selection:",
            placeholder="resname GLU",
            layout=Layout(height="auto", width="100%")
        )
        
        # Buttons
        self.buttons = {
            'add_p1': factory.create_simple_autobutton("Add mouse to P1", "info", "40%"),
            'add_p2': factory.create_simple_autobutton("Add mouse to P2", "info", "40%"),
            'add_sel_p1' : factory.create_simple_autobutton("Add selection to P1", "info", "40%"),
            'add_sel_p2' : factory.create_simple_autobutton("Add selection to P2", "info", "40%"),
            'pop_p1': factory.create_simple_autobutton("Pop P1", "info", "40%"),
            'pop_p2': factory.create_simple_autobutton("Pop P2", "info", "40%"),
            'clear_p1': factory.create_simple_autobutton("Clear P1", "info", "40%"),
            'clear_p2': factory.create_simple_autobutton("Clear P2", "info", "40%"),
            'switch': factory.create_simple_autobutton("Switch P1/P2", "info", "50%"),
            'update': factory.create_simple_autobutton("Update Funnel", "danger", "100%"),
            'surface': factory.create_simple_autobutton("Surface", "info", "10%")
        }
        
    def _setup_layout(self):
        """Organize widgets into a cohesive layout."""
        # Parameter section
        param_box = widgets.VBox([
            widgets.TwoByTwoLayout(
                **dict(
                    top_left=self.sliders['lower_wall'], 
                    top_right=self.sliders['upper_wall'],
                    bottom_left=self.sliders['wall_width'],
                    bottom_right=self.sliders['wall_buffer']
                )
            ),
            widgets.HBox([self.sliders['alpha'], self.sliders['s']])
        ])
        
        # Selection tools
        selection_box = widgets.VBox(
            [widgets.HBox([
                    self.selection_input,
                    self.buttons['add_p1'], self.buttons['clear_p1'],
                    self.buttons['add_p2'], self.buttons['clear_p2']]),
                widgets.HBox([
                    self.buttons['add_sel_p1'], self.buttons['pop_p1'],
                    self.buttons['add_sel_p2'], self.buttons['pop_p2']
                ])
            ]
        )
        
        
        # Control buttons
        control_box = widgets.HBox([
            self.buttons['switch'],
            self.buttons['update'],
            self.buttons['surface']
        ])

        def _print_funnel_parameters(p1, p2, lig, alpha, s, lower, upper, width, buffer):
            print(
                "P1: {}\nP2: {}\nlig: {}\nAlpha: {}, s: {}, lower_wall: {}, upper_wall: {}, wall_width: {}, wall_buffer {}".format(
                    p1, p2, lig, alpha, s, lower, upper, width, buffer
                )
            )
        output_text = widgets.VBox([
            widgets.interactive_output(
            _print_funnel_parameters,
            self.get_funnel_parameters())
            ])
        
        # Combine all sections
        self.layout = widgets.VBox(
            [
                 widgets.VBox([
                    param_box,
                    selection_box,
                    control_box,
                    output_text
                    
                 ],
                 layout=Layout(border="solid 2px")
                 ),
            self.viewer.view
        ])

        
    @staticmethod
    def _print_funnel_parameters(p1, p2, lig, alpha, s, lower, upper, width, buffer):
        print(
                "P1: {}\nP2: {}\nlig: {}\nAlpha: {}, s: {}, lower_wall: {}, upper_wall: {}, wall_width: {}, wall_buffer {}".format(
                    p1, p2, lig, alpha, s, lower, upper, width, buffer
                )
            )

        
    def _bind_callbacks(self):
        """Set up all widget callbacks."""
        # Button callbacks
        self.buttons['add_p1'].on_click(lambda _: self._add_selection('p1'))
        self.buttons['add_p2'].on_click(lambda _: self._add_selection('p2'))
        self.buttons['add_sel_p1'].on_click(lambda _: self._add_selection('p1'))
        self.buttons['add_sel_p2'].on_click(lambda _: self._add_selection('p2'))
        self.buttons['clear_p1'].on_click(lambda _: self._clear_points('p1'))
        self.buttons['clear_p2'].on_click(lambda _: self._clear_points('p2'))
        self.buttons['switch'].on_click(lambda _: self._switch_points())
        self.buttons['update'].on_click(lambda _: self._update_funnel())
        self.buttons['surface'].on_click(lambda _: self.viewer.toggle_surface())
        
        # Parameter update callbacks
        for name, slider in self.sliders.items():
            slider.observe(lambda change: self._update_parameter(name, change['new']), names='value')
        
    def _add_selection(self, point_type: str):
        """Add current selection to specified point group."""
        selection = self.selection_input.value
        try:
            indices = self.trajectory.top.select(selection).tolist()
            if indices:
                setattr(self.params, point_type, indices)
                self._update_point_visualization()
        except Exception as e:
            print(f"Selection error: {e}")
          
    def _clear_points(self, point_type: str):
        """Clear points of specified type."""
        setattr(self.params, point_type, [])
        self._update_point_visualization()
        
    def _switch_points(self):
        """Switch P1 and P2 points."""
        self.params.p1, self.params.p2 = self.params.p2, self.params.p1
        self._update_point_visualization()
        self._update_funnel()
        
    def _update_parameter(self, name: str, value: float):
        """Update a funnel parameter value."""
        setattr(self.params, name, value)
        
    def _update_point_visualization(self):
        """Update the visualization of P1 and P2 points."""
        # Clear existing visualizations
        self.viewer.view.shape.clear()
        
        # Add P1 points (red)
        if self.params.p1:
            coords = self._get_coordinates(self.params.p1)
            self.viewer.add_point_visualization(coords.mean(axis=0), [1, 0, 0], 1.0)
            for coord in coords:
                self.viewer.add_point_visualization(coord, [1, 0, 0], 0.4)
                
        # Add P2 points (green)
        if self.params.p2:
            coords = self._get_coordinates(self.params.p2)
            self.viewer.add_point_visualization(coords.mean(axis=0), [-1, 2, 0], 1.0)
            for coord in coords:
                self.viewer.add_point_visualization(coord, [-1, 2, 0], 0.4)
                
    def _update_funnel(self):
        """Update the funnel visualization."""
        if self.params.p1 and self.params.p2:
            # Create funnel potential with current parameters
            funnel = self.get_funnel_parameters()
            funnel_coords=self.params.compute_funnel_surface(funnel)
            # Update visualization
            self.viewer.update_funnel_visualization(funnel_coords)
            
    def _get_coordinates(self, indices: List[int]) -> np.ndarray:
        """Get coordinates for specified atom indices."""
        return self.trajectory.xyz[0][indices]
        
    def display(self):
        """Display the dashboard."""
        return self.layout
        
    def get_funnel_parameters(self) -> Dict[str, Any]:
        """Get current funnel parameters as a dictionary."""
        return {
            'p1': self._get_coordinates(self.params.p1).mean(axis=0).tolist() if self.params.p1 else None,
            'p2': self._get_coordinates(self.params.p2).mean(axis=0).tolist() if self.params.p2 else None,
            'lig': self.ligand_indices,
            'alpha': self.params.alpha,
            's': self.params.s,
            'lower_wall': self.params.lower_wall,
            'upper_wall': self.params.upper_wall,
            'wall_width': self.params.wall_width,
            'wall_buffer': self.params.wall_buffer
        }
        
    def get_funnel(self) -> Optional[SigmoidFunnelPotential]:
        """Get configured funnel potential object."""
        params = self.get_funnel_parameters()
        if None in [params['p1'], params['p2']]:
            return None
        return SigmoidFunnelPotential(**params)