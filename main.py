import sys

from PySide6.QtWidgets import QMainWindow, QApplication

from calculations_module import CalculationsModule
from design import Ui_MainWindow


class GeometryWizard(QMainWindow):
    def __init__(self):
        super(GeometryWizard, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(490, 340)

        self.calculationsModule = CalculationsModule()

        self.ui.solveBtn.clicked.connect(self.__run_calculations)

    def __run_calculations(self):
        self.calculationsModule.l_st = float(self.ui.lStText.text())
        self.calculationsModule.l_ot = float(self.ui.lOtTex.text())
        self.calculationsModule.r_vn = float(self.ui.rVnText.text())
        self.calculationsModule.t = float(self.ui.tText.text())

        self.calculationsModule.calculate_parameters()

        self.ui.rOsText.setText(f'{self.calculationsModule.r_os:.2f}')
        self.ui.aText.setText(f'{self.calculationsModule.a:.2f}')
        self.ui.sXText.setText(f'{self.calculationsModule.s_x:.2f}')
        self.ui.sYText.setText(f'{self.calculationsModule.s_y:.2f}')
        self.ui.xCText.setText(f'{self.calculationsModule.x_c:.2f}')
        self.ui.yCText.setText(f'{self.calculationsModule.y_c:.2f}')
        self.ui.jXText.setText(f'{self.calculationsModule.j_x:.2f}')
        self.ui.jYText.setText(f'{self.calculationsModule.j_y:.2f}')
        self.ui.jPText.setText(f'{self.calculationsModule.j_p:.2f}')
        self.ui.wXText.setText(f'{self.calculationsModule.w_x:.2f}')
        self.ui.wYText.setText(f'{self.calculationsModule.w_y:.2f}')
        self.ui.iXText.setText(f'{self.calculationsModule.i_x:.2f}')
        self.ui.iYText.setText(f'{self.calculationsModule.i_y:.2f}')
        self.ui.hText.setText(f'{self.calculationsModule.h:.2f}')
        self.ui.bText.setText(f'{self.calculationsModule.b:.2f}')


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GeometryWizard()
    window.show()

    sys.exit(app.exec())
