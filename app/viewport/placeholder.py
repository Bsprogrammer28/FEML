from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class ViewportPlaceholder(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("3D Viewport\n(Geometry & Results will appear here)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: gray;")
        layout.addWidget(label)