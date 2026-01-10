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

    def _create_viewport(self):
        # from viewport.vtk_view import BeamViewport
        # self.viewport = BeamViewport()
        # self.setCentralWidget(self.viewport)
        from viewport.placeholder import ViewportPlaceholder
        self.viewport = ViewportPlaceholder()
        self.setCentralWidget(self.viewport)

    def _create_docks(self):
        from docks.geometry_dock import GeometryDock
        from docks.load_dock import LoadDock
        from docks.material_dock import MaterialDock
        from docks.results_dock import ResultsDock

        self.geometry_dock = GeometryDock(self)
        self.load_dock = LoadDock(self)
        self.material_dock = MaterialDock(self)
        self.results_dock = ResultsDock(self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.geometry_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.load_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.material_dock)

        self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)

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
