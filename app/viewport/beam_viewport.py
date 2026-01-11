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
        self.clamp_actor = None
        self.load_actors = []
    
    def set_beam(self, beam):
        self.plotter.clear()
        self.beam_actor = self.plotter.add_mesh(
            beam, 
            color="lightgray",
            show_edges=True
        )
        self.plotter.reset_camera()
        self.plotter.render()
    
    def set_clamp(self, clamp):
        if self.clamp_actor:
            self.plotter.remove_actor(self.clamp_actor)
        self.clamp_actor = self.plotter.add_mesh(
            clamp,
            color="red",
            opacity=0.5
        )
        self.plotter.render()
    
    def set_loads(self, loads):
        for a in self.load_actors:
            self.plotter.remove_actor(a)
        self.load_actors.clear()

        for arrow in loads:
            actor = self.plotter.add_mesh(
                arrow,
                color="blue"
            )
            self.load_actors.append(actor)
        
        self.plotter.render()

    def show_results(self, mesh, displacement, field="Z Disp"):
        mesh = mesh.copy()
        mesh["displacement"] = displacement
        mesh[field] = displacement[:, 2]  

        warped = mesh.warp_by_vector("displacement", factor=1.0)

        self.plotter.clear()
        self.plotter.add_mesh(
            warped, 
            scalars=field,
            cmap="viridis",
            show_edges=True
        )

        self.plotter.show_axes()
        self.plotter.reset_camera()
        self.plotter.render()
        