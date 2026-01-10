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

def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    from main_window import MainWindow

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()