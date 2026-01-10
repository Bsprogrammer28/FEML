from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox
from PyQt5.QtCore import Qt

class GeometryDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Geometry", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.length = QDoubleSpinBox()
        self.length.setValue(2.0)
        self.length.setSuffix(" m")

        self.width = QDoubleSpinBox()
        self.width.setValue(0.1)
        self.width.setSuffix(" m")

        self.height = QDoubleSpinBox()
        self.height.setValue(0.05)
        self.height.setSuffix(" m")

        layout.addWidget(QLabel("Length:"))
        layout.addWidget(self.length)
        layout.addWidget(QLabel("Width:"))
        layout.addWidget(self.width)
        layout.addWidget(QLabel("Height:"))
        layout.addWidget(self.height)

        layout.addStretch()
        self.setWidget(widget)