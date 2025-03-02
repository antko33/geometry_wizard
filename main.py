import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from design import Ui_MainWindow


class GeometryWizard(QMainWindow):
    def __init__(self):
        super(GeometryWizard, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setCentralWidget(self.ui.tabWidget)
        self.setFixedSize(600, 550)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GeometryWizard()
    window.show()

    sys.exit(app.exec())
