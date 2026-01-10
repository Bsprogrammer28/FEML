from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyvista as pv

class BeamViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        self.plotter.set_background("white")
        self.plotter.show_axes()

        self.beam_actor = None

    def show_beam(self, mesh):
        self.plotter.clear()
        self.beam_actor = self.plotter.add_mesh(
            mesh, 
            color="lightgray",
            show_edges=True
        )
        self.plotter.reset_camera()