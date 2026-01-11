from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout,
    QLabel, QPushButton, QComboBox
)
from PyQt5.QtCore import Qt

class ResultsDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Results", main_window)
        self.main_window = main_window
        self.setAllowedAreas(Qt.RightDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.result_type = QComboBox()
        self.result_type.addItems([
            "Z Displacement",
        ])

        solve_btn = QPushButton("Solve")
        solve_btn.clicked.connect(self.run_preview)

        layout.addWidget(QLabel("Result Type:"))
        layout.addWidget(self.result_type)
        layout.addWidget(solve_btn)
        layout.addStretch()

        self.setWidget(widget)

    def run_preview(self):
        self.main_window.run_preview_results()