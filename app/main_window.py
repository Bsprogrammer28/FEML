from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt
import torch
import numpy as np
from src.pinn_interface import Beam3DPINN

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN Beam Solver")
        self.resize(1400, 900)

        self._create_viewport()
        self._create_docks()

        from src.pinn_interface import Beam3DPINN
        self.pinn_model = Beam3DPINN(
            model_path="models/beam_model3d.pth",
            device="cuda" if torch.cuda.is_available() else "cpu")
        
        self._create_menu()
        self.update_load_preview()

    def _create_viewport(self):
        from viewport.beam_viewport import BeamViewport
        self.viewport = BeamViewport()
        self.setCentralWidget(self.viewport)

    def update_load_preview(self):
        from viewport.geometry_factory import (
            create_clamp_face,
            create_force_arrow,
        )

        W = self.geometry_dock.width.value()
        H = self.geometry_dock.height.value()
        L = self.geometry_dock.length.value()

        clamp = create_clamp_face(W, H)

        dir_map = {
            "+X": (1, 0, 0),
            "-X": (-1, 0, 0),
            "+Y": (0, 1, 0),
            "-Y": (0, -1, 0),
            "+Z": (0, 0, 1),
            "-Z": (0, 0, -1),
        }

        ld = self.load_dock
        direction = dir_map[ld.direction.currentText()]

        xn = ld.x.value()
        yn = ld.y.value()
        zn = ld.z.value()

        x_phys = xn * L
        y_phys = (yn - 0.5) * W
        z_phys = (zn - 0.5) * H

        arrow = create_force_arrow(
            origin=(x_phys, y_phys, z_phys),        # force visible location
            direction=direction,
            scale=0.15 * L
        )

        self.viewport.set_clamp(clamp)
        self.viewport.set_loads([arrow])

    def _create_docks(self):
        from docks.geometry_dock import GeometryDock
        from docks.load_dock import LoadDock
        from docks.results_dock import ResultsDock
        from docks.material_dock import MaterialDock

        self.geometry_dock = GeometryDock(self)
        self.load_dock = LoadDock(self)
        self.material_dock = MaterialDock(self)
        self.results_dock = ResultsDock(self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.geometry_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.load_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.material_dock)

        self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)

    def run_preview_results(self):
        from viewport.geometry_factory import create_beam, sample_mesh

        L = self.geometry_dock.length.value()
        W = self.geometry_dock.width.value()
        H = self.geometry_dock.height.value()

        beam = create_beam(L, W, H)
        sampled = sample_mesh(beam, 6000)

        x = sampled.points[:, 0]/L
        y = sampled.points[:, 1]/L
        z = sampled.points[:, 2]/L

        pts = np.column_stack([x, y, z])

        disp = self.pinn_model.predict(pts)

        self.viewport.show_results(sampled, disp)

    def _create_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction("New Project")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        solve_menu = menu.addMenu("Solve")
        solve_menu.addAction("Run Solver")

        view_menu = menu.addMenu("View")
        view_menu.addAction("Reset View")
