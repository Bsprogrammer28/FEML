from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox
from PyQt5.QtCore import Qt

class GeometryDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Geometry", main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.LeftDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.length = QDoubleSpinBox()
        self.length.setRange(0.1, 100.0)
        self.length.setValue(2.0)
        self.length.setSuffix(" m")

        self.width = QDoubleSpinBox()
        self.width.setRange(0.01, 10.0)
        self.width.setValue(0.1)
        self.width.setSuffix(" m")

        self.height = QDoubleSpinBox()
        self.height.setRange(0.01, 10.0)
        self.height.setValue(0.05)
        self.height.setSuffix(" m")

        for label, box in [
            ("Length", self.length),
            ("Width", self.width),
            ("Height", self.height),
        ]:
            layout.addWidget(QLabel(label))
            layout.addWidget(box)
            box.valueChanged.connect(self.update_geometry)
        
        layout.addStretch()
        self.setWidget(widget)

        self.update_geometry()

    def update_geometry(self):
        from viewport.geometry_factory import create_beam

        L = self.length.value()
        W = self.width.value()
        H = self.height.value()

        beam_mesh = create_beam(L, W, H)
        self.main_window.viewport.set_beam(beam_mesh)