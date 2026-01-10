# Windows GUI FIX
import os
import platform
if platform.system() == "Windows":
    import ctypes
from importlib.util import find_spec
try:
    if (spec := find_spec("torch")) and spec.origin and os.path.exists(
    dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
    ):
        ctypes.CDLL(os.path.normpath(dll_path))
except Exception:
    pass

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton
import sys
import geometryPreprocessor

class BeamGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Geometry Input")

        layout = QVBoxLayout()

        self.length = QDoubleSpinBox()
        self.length.setValue(2.0)
        self.length.setSuffix(" m")

        self.width = QDoubleSpinBox()
        self.width.setValue(0.1)

        self.height = QDoubleSpinBox()
        self.height.setValue(0.05)

        btn = QPushButton("Generate PINN Input")
        btn.clicked.connect(self.generate)

        layout.addWidget(QLabel("Length:"))
        layout.addWidget(self.length)
        layout.addWidget(QLabel("Width:"))
        layout.addWidget(self.width)
        layout.addWidget(QLabel("Height:"))
        layout.addWidget(self.height)
        layout.addWidget(btn)

        self.setLayout(layout)

    def generate(self):
        L = self.length.value()
        W = self.width.value()
        H = self.height.value()
        Lx, Ly, Lz = geometryPreprocessor.normalize_geometry(L, W, H)
        print("Scalling: ", Lx, Ly, Lz)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = BeamGUI()
    gui.show()
    sys.exit(app.exec_())
