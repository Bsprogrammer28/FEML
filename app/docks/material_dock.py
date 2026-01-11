from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox
from PyQt5.QtCore import Qt

class MaterialDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Material Properties", main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.LeftDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.material_type = QComboBox()
        self.material_type.addItems([
            "Linear Elastic",
        ])

        self.E = QDoubleSpinBox()
        self.E.setRange(1e3, 1e12)
        self.E.setValue(210e9)  # Default to steel
        self.E.setSuffix(" Pa")
        self.E.setDecimals(3)

        self.nu = QDoubleSpinBox()
        self.nu.setRange(0.0, 0.5)
        self.nu.setValue(0.3)  # Default to steel
        self.nu.setDecimals(4)
        self.nu.setSingleStep(0.01)

        layout.addWidget(QLabel("Material Type:"))
        layout.addWidget(self.material_type)
        layout.addWidget(QLabel("Young's Modulus (E):"))
        layout.addWidget(self.E)
        layout.addWidget(QLabel("Poisson's Ratio (ν):"))
        layout.addWidget(self.nu)
        layout.addStretch()

        self.setWidget(widget)

    def get_material(self):
        E = self.E.value()
        nu = self.nu.value()

        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        return {
            "model": "linear_elastic",
            "E": E,
            "nu": nu,
            "mu": mu,
            "lambda": lam
        }