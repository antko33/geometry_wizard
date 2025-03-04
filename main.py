import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from design import Ui_MainWindow
from main_viewmodel import MainViewModel


class GeometryWizard(QMainWindow):
    def __init__(self):
        super(GeometryWizard, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setCentralWidget(self.ui.tabWidget)
        self.setFixedSize(600, 550)

        self.viewModel = MainViewModel()

        self.ui.solveBtn.clicked.connect(self.__run_calculations)

    def __run_calculations(self):
        self.viewModel.l_st = float(self.ui.lStText.text())
        self.viewModel.l_ot = float(self.ui.iOtText.text())  # TODO: Typo
        self.viewModel.r_vn = float(self.ui.rVnText.text())
        self.viewModel.t = float(self.ui.tText.text())

        self.viewModel.calculate_parameters()

        self.ui.rOsText.setText(str(self.viewModel.r_os))
        self.ui.aText.setText(str(self.viewModel.a))
        self.ui.sXText.setText(str(self.viewModel.s_x))
        self.ui.sYText.setText(str(self.viewModel.s_y))
        self.ui.xCText.setText(str(self.viewModel.x_c))
        self.ui.yCText.setText(str(self.viewModel.y_c))
        self.ui.jXText.setText(str(self.viewModel.j_x))
        self.ui.jYText.setText(str(self.viewModel.j_y))
        self.ui.jPText.setText(str(self.viewModel.j_p))
        self.ui.wXText.setText(str(self.viewModel.w_x))
        self.ui.wYText.setText(str(self.viewModel.w_y))
        self.ui.iXText.setText(str(self.viewModel.i_x))
        self.ui.iYText.setText(str(self.viewModel.i_y))
        self.ui.hText.setText(str(self.viewModel.h))
        self.ui.bText.setText(str(self.viewModel.b))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GeometryWizard()
    window.show()

    sys.exit(app.exec())
