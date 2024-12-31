from typing import Optional

from ipywidgets import TwoByTwoLayout
from ipywidgets import VBox
from ipywidgets import HBox
from ipywidgets import Button
from ipywidgets import FloatSlider
from ipywidgets import Layout

import nglview as nv
import pandas as pd
import mdtraj as md
import numpy as np
import ipywidgets as widgets



from funnel.funnels import SigmoidFunnelPotential

def create_simple_autobutton(description, button_style, width="auto", b_type=Button):
    return b_type(
        description=description, button_style=button_style, layout=Layout(height="auto", width=width)
    )

class FunnelDashboard:
    """
    Create an interactive IPyWidget dashboard for the funnel creation.
    Example to use it:
    dashboard=FunnelDashBoard(trajectory)
    dashboard.display()

    # To get the funnel as a BiasVariable
    dashboard.get_funnel()

    # To get the dictionary of the parameters of the funnel
    dashboard.get_funnel_porameters()
    """

    def __init__(self, trajectory: md.Trajectory, ligand_name: Optional[str] = None):
        """
        Args:
            trajectory: trajectory to use for the dashboard
            ligand_name: name of the ligand to get the indexs, if None is given the names used to automatically
                        search for it will be "SLT", "MOL" and "UNK"
        """
        self.trajectory = trajectory
        self.sliders1, self.sliders2, self.selection1, self.selection2, self.upd = self.create_layout()
        self.p1_index = []
        self.p1_coord = np.array([20, 20, 20])
        self.p2_index = []
        self.p2_coord = np.array([-5, -4, -4.2])
        self.p1_selection = []
        self.p2_selection = []
        self.ligand_name = ligand_name
        self.ligang_index = self.get_ligand()
        self.com1_coord = []
        self.com2_coord = []

        self.p1 = widgets.Text(str(self.p1_index))
        self.p2 = widgets.Text(str(self.p2_index))
        self.alpha = self.sliders2.children[0]
        self.s = self.sliders2.children[1]
        self.lower_wall = self.sliders1.children[0]
        self.upper_wall = self.sliders1.children[1]
        self.wall_width = self.sliders1.children[2]
        self.wall_buffer = self.sliders1.children[3]
        self.first_run = True
        self.surface = False
        self.created = False
        self.create_view()

    def search_resname(self, name: str):
        return self.trajectory.top.select(f"resname {name}").tolist()

    def get_ligand(self):
        if self.ligand_name is None:
            possible_names = ["SLT", "MOL", "UNK"]
            for name in possible_names:
                idx = self.search_resname(name)
                if len(idx) > 0:
                    return idx
        else:
            idx = self.search_resname(self.ligand_name)
            if len(idx) > 0:
                return idx
        return []

    def create_layout(self):
        a = TwoByTwoLayout(
            top_left=FloatSlider(
                description="lower_wall",
                value=0.0,
                min=-5.00,
                max=5.00,
                step=0.1,
                layout=Layout(height="auto", width="auto"),
            ),
            top_right=FloatSlider(
                description="upper_wall",
                value=3.0,
                min=0.10,
                max=15.0,
                step=0.1,
                layout=Layout(height="auto", width="auto"),
            ),
            bottom_left=FloatSlider(
                description="wall_width",
                value=0.5,
                min=0.0,
                max=5.0,
                step=0.01,
                layout=Layout(height="auto", width="auto"),
            ),
            bottom_right=FloatSlider(
                description="wall_buffer",
                value=0.1,
                min=0.0,
                max=5.0,
                step=0.01,
                layout=Layout(height="auto", width="auto"),
            ),
        )
        b = HBox(
            [
                FloatSlider(
                    description="alpha",
                    value=3.0,
                    min=0.0,
                    max=30.0,
                    step=0.1,
                    layout=Layout(height="auto", width="100%"),
                ),
                FloatSlider(
                    description="s",
                    value=0.1,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    layout=Layout(height="auto", width="100%"),
                ),
            ]
        )
        c = HBox(
            [
                widgets.Text(
                    description="Selection:",
                    placeholder="resname GLU",
                    layout=Layout(height="auto", width="100%"),
                ),
                create_simple_autobutton("Add selection to P1", "info", "40%", Button),
                create_simple_autobutton("Remove last from P1", "info", "40%", Button),
                create_simple_autobutton("Clear P1", "info", "40%", Button),
            ]
        )
        d = HBox(
            [
                create_simple_autobutton("Switch", "info", "50%", Button),
                create_simple_autobutton("Add mouse to P1", "info", "40%", Button),
                create_simple_autobutton("Add mouse to P2", "info", "40%", Button),
                create_simple_autobutton("Add selection to P2", "info", "40%", Button),
                create_simple_autobutton("Remove last from P2", "info", "40%", Button),
                create_simple_autobutton("Clear P2", "info", "40%", Button),
            ]
        )
        e = HBox(
            [
                create_simple_autobutton("Update Funnel", "danger", "100%", Button),
                create_simple_autobutton("Surface", "info", "10%", Button),
            ]
        )

        # Define the button callbacks
        c.children[1].on_click(
            lambda x: [
                self.p1_selection.append(c.children[0].value),
                self.extract_index(),
                self.clear_selection(),
                self.p1_selection.clear(),
            ]
        )
        c.children[2].on_click(lambda x: [self.p1_index.pop(), self.create_balls()])
        c.children[3].on_click(lambda x: [self.p1_index.clear(), self.p1_coord.clear(), self.create_balls()])

        d.children[0].on_click(lambda x: self.pressed_switch())
        d.children[1].on_click(lambda x: self.add_to_p1())
        d.children[2].on_click(lambda x: self.add_to_p2())
        d.children[3].on_click(
            lambda x: [
                self.p2_selection.append(c.children[0].value),
                self.extract_index(),
                self.clear_selection(),
                self.p1_selection.clear(),
            ]
        )
        d.children[4].on_click(lambda x: [self.p2_index.pop(), self.create_balls()])
        d.children[5].on_click(lambda x: [self.p2_index.clear(), self.p2_coord.clear(), self.create_balls()])

        e.children[0].on_click(lambda x: self.update_funnel_viz())
        e.children[1].on_click(lambda x: self.change_view())

        return a, b, c, d, e

    def create_output(self):
        def f(p1, p2, alpha, s, lower, upper, width, buffer):
            print(
                "P1: {}\nP2: {}\nAlpha: {}, s: {}, lower_wall: {}, upper_wall: {}, wall_width: {}, wall_buffer {}".format(
                    p1, p2, alpha, s, lower, upper, width, buffer
                )
            )

        out = widgets.interactive_output(
            f,
            {
                "p1": self.p1,
                "p2": self.p2,
                "alpha": self.alpha,
                "s": self.s,
                "lower": self.lower_wall,
                "upper": self.upper_wall,
                "width": self.wall_width,
                "buffer": self.wall_buffer,
            },
        )
        return VBox([out])

    @property
    def get_mouse_sel(self):
        return self.view.get_state()["picked"]

    @property
    def get_index(self):
        try:
            return self.get_mouse_sel["atom1"]["index"]
        except:
            return None

    def extract_index(self):
        if len(self.p1_selection) > 0:
            sel_string = self.p1_selection[0]
            try:
                selection = self.trajectory.top.select(sel_string)
                if selection.size > 0:
                    self.p1_index.extend(selection.tolist())
            except:
                pass

        if len(self.p2_selection) > 0:
            sel_string = self.p2_selection[0]
            try:
                selection = self.trajectory.top.select(sel_string)
                if selection.size > 0:
                    self.p2_index.extend(selection.tolist())
            except:
                pass
        self.create_balls()

    def clear_selection(self):
        self.selection1.children[0].value = ""

    def change_view(self):
        if not self.surface:
            self.view.add_surface(selection="protein", opacity=0.4, color="gray")
            self.surface = True

        else:
            self.view.clear_representations()
            self.view.add_trajectory(self.trajectory)
            self.view.add_cartoon("protein")
            self.view.add_ball_and_stick("ligand")
            self.view.center()
            self.view.background = "black"
            self.view.render_image(trim=True, factor=2)
            self.surface = False

    def add_to_p1(self):
        index = self.get_index
        if index is None:
            pass
        else:
            self.p1_index.append(index)
            self.create_balls()

    def add_to_p2(self):
        index = self.get_index
        if index is None:
            pass
        else:
            self.p2_index.append(index)
            self.create_balls()

    def pressed_switch(self):
        p1_index = self.p1_index
        p1_coord = self.p1_coord
        self.p1_index = self.p2_index
        self.p1_coord = self.p2_coord
        self.p2_index = p1_index
        self.p2_coord = p1_coord
        self.update_funnel_viz()
        self.create_balls()

    def create_view(self):
        view = nv.NGLWidget(gui=True, height="500px")
        view.add_trajectory(self.trajectory)
        view.add_cartoon("protein")
        view.add_principal_axes("ligand")
        view.center()
        view.background = "black"
        view.render_image(trim=True, factor=2)
        view.add_trajectory(self.make_funnel())
        self.view = view
        self.first_run = False

    def display(self):
        """
        Display the widget
        """
        a = VBox(
            [
                VBox(
                    [
                        self.sliders1,
                        self.sliders2,
                        self.selection1,
                        self.selection2,
                        self.upd,
                        self.create_output(),
                    ],
                    layout=Layout(border="solid 2px"),
                ),
                self.view,
            ]
        )
        return a

    def get_coms(self):
        if len(self.p1_index) > 0:
            com1 = np.atleast_2d(self.xyz(self.p1_index)).mean(axis=0)
        if len(self.p1_index) == 0:
            com1 = np.array([0, 0, 0])
        if len(self.p2_index) > 0:
            com2 = np.atleast_2d(self.xyz(self.p2_index)).mean(axis=0)
        if len(self.p2_index) == 0:
            com2 = np.array([3, 1, 3])
        return com1, com2

    def compute_funnel_surface(self):
        com1, com2 = self.get_coms()

        lower_wall = self.lower_wall.value
        upper_wall = self.upper_wall.value
        wall_width = self.wall_width.value
        beta_cent = self.alpha.value
        s_cent = self.s.value
        wall_buffer = self.wall_buffer.value

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

    def make_funnel(self):
        return self.make_funnel_trajectory(self.compute_funnel_surface())

    @staticmethod
    def make_funnel_trajectory(funnel: np.ndarray) -> md.Trajectory:
        """
        Create a dummy mdtraj trajectory from the funnel surface coordinate.
        """
        df = pd.DataFrame(columns=["serial", "name", "element", "resSeq", "resName", "chainID"])
        for i in range(len(funnel)):
            df.loc[i] = [i, f"HE{i + 1}", "He", 1, "FNL", 0]
        return md.Trajectory(xyz=funnel, topology=md.Topology().from_dataframe(df))

    def clean_components(self):
        # really ugly but I can't find any way to access a container of components
        self.view.remove_component(self.view.component_2)  # ignore
        self.view.remove_component(self.view.component_3)
        self.view.remove_component(self.view.component_4)
        self.view.remove_component(self.view.component_5)
        self.view.remove_component(self.view.component_6)
        self.view.remove_component(self.view.component_7)
        self.view.remove_component(self.view.component_8)
        self.view.remove_component(self.view.component_9)
        self.view.remove_component(self.view.component_10)
        self.view.remove_component(self.view.component_11)
        self.view.remove_component(self.view.component_12)
        self.view.remove_component(self.view.component_13)
        self.view.remove_component(self.view.component_14)
        self.view.remove_component(self.view.component_15)
        self.view.remove_component(self.view.component_16)
        self.view.remove_component(self.view.component_17)
        self.view.remove_component(self.view.component_18)
        self.view.remove_component(self.view.component_19)
        self.view.remove_component(self.view.component_20)
        self.view.remove_component(self.view.component_21)
        self.view.remove_component(self.view.component_22)
        self.view.remove_component(self.view.component_23)
        self.view.remove_component(self.view.component_24)
        self.view.remove_component(self.view.component_25)
        self.view.remove_component(self.view.component_26)
        self.view.remove_component(self.view.component_27)
        self.view.remove_component(self.view.component_28)
        self.view.remove_component(self.view.component_29)
        self.view.remove_component(self.view.component_30)

    def update_funnel_viz(self):
        try:
            self.view.remove_component(self.view.component_1)
            self.clean_components()
        except Exception:
            pass
        self.create_balls()
        self.view.add_trajectory(self.make_funnel())

    def create_balls(self):
        try:
            self.clean_components()
        except Exception:
            pass
        if len(self.p1_index) > 0:
            self.add_p1_balls()
        if len(self.p2_index) > 0:
            self.add_p2_balls()

        self.p1.value = str(self.p1_index)
        self.p2.value = str(self.p2_index)

    def add_p1_balls(self):
        coord = self.xyz(self.p1_index)
        for i in coord:
            self.view.shape.add_sphere(i * 10, [1, 0, 0], 0.4)
        self.view.shape.add_sphere(coord.mean(axis=0) * 10, [1, 0, 0], 1)

    def add_p2_balls(self):
        coord = self.xyz(self.p2_index)
        for i in coord:
            self.view.shape.add_sphere(i * 10, [-1, 2, 0], 0.4)
        self.view.shape.add_sphere(coord.mean(axis=0) * 10, [-1, 2, 0], 1)

    def xyz(self, index) -> np.ndarray:
        return self.trajectory.xyz[0][index]

    def get_funnel_parameters(self) -> dict:
        """
        Get the parameters for the funnel as a dictionary
        """
        datum = {
            "p1": self.p1_index,
            "p2": self.p2_index,
            "lig": self.ligang_index,
            "alpha": self.alpha.value,
            "s": self.s.value,
            "lower_wall": self.lower_wall.value,
            "upper_wall": self.upper_wall.value,
            "wall_width": self.wall_width.value,
            "wall_buffer": self.wall_buffer.value,
        }
        return datum

    def get_funnel(self) -> SigmoidFunnelPotential:
        """
        Get the funnel potential object
        Returns:
            SigmoidFunnelPotential
        """
        return SigmoidFunnelPotential(**self.get_funnel_parameters())
