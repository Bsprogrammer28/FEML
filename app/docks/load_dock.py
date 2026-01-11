from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QComboBox, QDoubleSpinBox
from PyQt5.QtCore import Qt

class LoadDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Load Settings", main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.LeftDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.direction = QComboBox()
        self.direction.addItems(["+X", "-X", "+Y", "-Y", "+Z", "-Z"])

        self.magnitude = QDoubleSpinBox()
        self.magnitude.setRange(0, 1e6)
        self.magnitude.setValue(1000)
        self.magnitude.setSuffix(" N")

        self.x = QDoubleSpinBox()
        self.y = QDoubleSpinBox()
        self.z = QDoubleSpinBox()

        for label, box in [
            ("Direction:", self.direction),
            ("Magnitude:", self.magnitude),
            ("X Position:", self.x),
            ("Y Position:", self.y),
            ("Z Position:", self.z),
        ]:
            layout.addWidget(QLabel(label))
            layout.addWidget(box)
        
        for w in [self.direction, self.magnitude, self.x, self.y, self.z]:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self.update_load)
            else:
                w.currentIndexChanged.connect(self.update_load)
        
        layout.addStretch()
        self.setWidget(widget)

        # self.update_load()

    def update_load(self):
        self.main_window.update_load_preview()

