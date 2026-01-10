from PyQt5.QtWidgets import QMainWindow, QApplication, QDockWidget
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN Beam Solver")
        self.resize(1400, 900)

        self._create_viewport()
        self._create_docks()
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

        arrow = create_force_arrow(
            origin=(L, 0, 0),        # force visible location
            direction=direction,
            scale=0.15 * L
        )

        self.viewport.set_clamp(clamp)
        self.viewport.set_loads([arrow])


    def _create_docks(self):
        from docks.geometry_dock import GeometryDock
        from docks.load_dock import LoadDock
        # from docks.material_dock import MaterialDock
        # from docks.results_dock import ResultsDock

        self.geometry_dock = GeometryDock(self)
        self.load_dock = LoadDock(self)
        # self.material_dock = MaterialDock(self)
        # self.results_dock = ResultsDock(self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.geometry_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.load_dock)
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.material_dock)

        # self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)

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
