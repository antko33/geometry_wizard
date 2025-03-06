import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from design import Ui_MainWindow
from main_viewmodel import MainViewModel


class GeometryWizard(QMainWindow):
    def __init__(self):
        super(GeometryWizard, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(490, 340)

        self.viewModel = MainViewModel()

        self.ui.solveBtn.clicked.connect(self.__run_calculations)

    def __run_calculations(self):
        self.viewModel.l_st = float(self.ui.lStText.text())
        self.viewModel.l_ot = float(self.ui.lOtTex.text())
        self.viewModel.r_vn = float(self.ui.rVnText.text())
        self.viewModel.t = float(self.ui.tText.text())

        self.viewModel.calculate_parameters()

        self.ui.rOsText.setText(f'{self.viewModel.r_os:.4f}')
        self.ui.aText.setText(f'{self.viewModel.a:.4f}')
        self.ui.sXText.setText(f'{self.viewModel.s_x:.4f}')
        self.ui.sYText.setText(f'{self.viewModel.s_y:.4f}')
        self.ui.xCText.setText(f'{self.viewModel.x_c:.4f}')
        self.ui.yCText.setText(f'{self.viewModel.y_c:.4f}')
        self.ui.jXText.setText(f'{self.viewModel.j_x:.4f}')
        self.ui.jYText.setText(f'{self.viewModel.j_y:.4f}')
        self.ui.jPText.setText(f'{self.viewModel.j_p:.4f}')
        self.ui.wXText.setText(f'{self.viewModel.w_x:.4f}')
        self.ui.wYText.setText(f'{self.viewModel.w_y:.4f}')
        self.ui.iXText.setText(f'{self.viewModel.i_x:.4f}')
        self.ui.iYText.setText(f'{self.viewModel.i_y:.4f}')
        self.ui.hText.setText(f'{self.viewModel.h:.4f}')
        self.ui.bText.setText(f'{self.viewModel.b:.4f}')


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GeometryWizard()
    window.show()

    sys.exit(app.exec())
