################################################################################
## Form generated from reading UI file 'design.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect)
from PySide6.QtWidgets import (QLabel, QLineEdit, QPushButton, QSizePolicy, QStatusBar, QTabWidget,
                               QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(600, 550)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(10, 0, 591, 521))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy1)
        self.inputTab = QWidget()
        self.inputTab.setObjectName(u"inputTab")
        self.label_8 = QLabel(self.inputTab)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(400, 250, 21, 16))
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_4 = QLabel(self.inputTab)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(400, 190, 21, 16))
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.tText = QLineEdit(self.inputTab)
        self.tText.setObjectName(u"tText")
        self.tText.setGeometry(QRect(430, 160, 113, 21))
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.tText.sizePolicy().hasHeightForWidth())
        self.tText.setSizePolicy(sizePolicy2)
        self.iOtText = QLineEdit(self.inputTab)
        self.iOtText.setObjectName(u"iOtText")
        self.iOtText.setGeometry(QRect(430, 220, 113, 21))
        sizePolicy2.setHeightForWidth(self.iOtText.sizePolicy().hasHeightForWidth())
        self.iOtText.setSizePolicy(sizePolicy2)
        self.label_5 = QLabel(self.inputTab)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(550, 220, 49, 16))
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_3 = QLabel(self.inputTab)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(550, 190, 49, 16))
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.rVnText = QLineEdit(self.inputTab)
        self.rVnText.setObjectName(u"rVnText")
        self.rVnText.setGeometry(QRect(430, 250, 113, 21))
        sizePolicy2.setHeightForWidth(self.rVnText.sizePolicy().hasHeightForWidth())
        self.rVnText.setSizePolicy(sizePolicy2)
        self.lStText = QLineEdit(self.inputTab)
        self.lStText.setObjectName(u"lStText")
        self.lStText.setGeometry(QRect(430, 190, 113, 21))
        sizePolicy2.setHeightForWidth(self.lStText.sizePolicy().hasHeightForWidth())
        self.lStText.setSizePolicy(sizePolicy2)
        self.label_2 = QLabel(self.inputTab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(550, 160, 49, 16))
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_7 = QLabel(self.inputTab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(550, 250, 49, 16))
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_6 = QLabel(self.inputTab)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(400, 220, 21, 16))
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label = QLabel(self.inputTab)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(400, 160, 21, 16))
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.solveBtn = QPushButton(self.inputTab)
        self.solveBtn.setObjectName(u"solveBtn")
        self.solveBtn.setGeometry(QRect(400, 300, 171, 51))
        self.tabWidget.addTab(self.inputTab, "")
        self.solution_tab = QWidget()
        self.solution_tab.setObjectName(u"solution_tab")
        self.label_9 = QLabel(self.solution_tab)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(540, 0, 49, 16))
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_10 = QLabel(self.solution_tab)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(390, 0, 21, 16))
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.rOsText = QLineEdit(self.solution_tab)
        self.rOsText.setObjectName(u"rOsText")
        self.rOsText.setGeometry(QRect(420, 0, 113, 21))
        sizePolicy2.setHeightForWidth(self.rOsText.sizePolicy().hasHeightForWidth())
        self.rOsText.setSizePolicy(sizePolicy2)
        self.label_11 = QLabel(self.solution_tab)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(540, 20, 49, 16))
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_12 = QLabel(self.solution_tab)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(390, 20, 21, 16))
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.aText = QLineEdit(self.solution_tab)
        self.aText.setObjectName(u"aText")
        self.aText.setGeometry(QRect(420, 20, 113, 21))
        sizePolicy2.setHeightForWidth(self.aText.sizePolicy().hasHeightForWidth())
        self.aText.setSizePolicy(sizePolicy2)
        self.label_13 = QLabel(self.solution_tab)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(540, 40, 49, 16))
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_14 = QLabel(self.solution_tab)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(390, 40, 21, 16))
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.sXText = QLineEdit(self.solution_tab)
        self.sXText.setObjectName(u"sXText")
        self.sXText.setGeometry(QRect(420, 40, 113, 21))
        sizePolicy2.setHeightForWidth(self.sXText.sizePolicy().hasHeightForWidth())
        self.sXText.setSizePolicy(sizePolicy2)
        self.label_15 = QLabel(self.solution_tab)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(540, 60, 49, 16))
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_16 = QLabel(self.solution_tab)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(390, 60, 21, 16))
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.sYText = QLineEdit(self.solution_tab)
        self.sYText.setObjectName(u"sYText")
        self.sYText.setGeometry(QRect(420, 60, 113, 21))
        sizePolicy2.setHeightForWidth(self.sYText.sizePolicy().hasHeightForWidth())
        self.sYText.setSizePolicy(sizePolicy2)
        self.label_17 = QLabel(self.solution_tab)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(540, 80, 49, 16))
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_18 = QLabel(self.solution_tab)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(390, 80, 21, 16))
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.xCText = QLineEdit(self.solution_tab)
        self.xCText.setObjectName(u"xCText")
        self.xCText.setGeometry(QRect(420, 80, 113, 21))
        sizePolicy2.setHeightForWidth(self.xCText.sizePolicy().hasHeightForWidth())
        self.xCText.setSizePolicy(sizePolicy2)
        self.label_19 = QLabel(self.solution_tab)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(540, 100, 49, 16))
        sizePolicy.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy)
        self.label_20 = QLabel(self.solution_tab)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(390, 100, 21, 16))
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.yCText = QLineEdit(self.solution_tab)
        self.yCText.setObjectName(u"yCText")
        self.yCText.setGeometry(QRect(420, 100, 113, 21))
        sizePolicy2.setHeightForWidth(self.yCText.sizePolicy().hasHeightForWidth())
        self.yCText.setSizePolicy(sizePolicy2)
        self.label_21 = QLabel(self.solution_tab)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(540, 120, 49, 16))
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_22 = QLabel(self.solution_tab)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(390, 120, 21, 16))
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.jXText = QLineEdit(self.solution_tab)
        self.jXText.setObjectName(u"jXText")
        self.jXText.setGeometry(QRect(420, 120, 113, 21))
        sizePolicy2.setHeightForWidth(self.jXText.sizePolicy().hasHeightForWidth())
        self.jXText.setSizePolicy(sizePolicy2)
        self.hText = QLineEdit(self.solution_tab)
        self.hText.setObjectName(u"hText")
        self.hText.setGeometry(QRect(420, 260, 113, 21))
        sizePolicy2.setHeightForWidth(self.hText.sizePolicy().hasHeightForWidth())
        self.hText.setSizePolicy(sizePolicy2)
        self.label_23 = QLabel(self.solution_tab)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(540, 180, 49, 16))
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.jYText = QLineEdit(self.solution_tab)
        self.jYText.setObjectName(u"jYText")
        self.jYText.setGeometry(QRect(420, 140, 113, 21))
        sizePolicy2.setHeightForWidth(self.jYText.sizePolicy().hasHeightForWidth())
        self.jYText.setSizePolicy(sizePolicy2)
        self.label_24 = QLabel(self.solution_tab)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(390, 200, 21, 16))
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.iXText = QLineEdit(self.solution_tab)
        self.iXText.setObjectName(u"iXText")
        self.iXText.setGeometry(QRect(420, 220, 113, 21))
        sizePolicy2.setHeightForWidth(self.iXText.sizePolicy().hasHeightForWidth())
        self.iXText.setSizePolicy(sizePolicy2)
        self.label_25 = QLabel(self.solution_tab)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(390, 260, 21, 16))
        sizePolicy.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy)
        self.label_26 = QLabel(self.solution_tab)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(390, 140, 21, 16))
        sizePolicy.setHeightForWidth(self.label_26.sizePolicy().hasHeightForWidth())
        self.label_26.setSizePolicy(sizePolicy)
        self.jPText = QLineEdit(self.solution_tab)
        self.jPText.setObjectName(u"jPText")
        self.jPText.setGeometry(QRect(420, 160, 113, 21))
        sizePolicy2.setHeightForWidth(self.jPText.sizePolicy().hasHeightForWidth())
        self.jPText.setSizePolicy(sizePolicy2)
        self.label_27 = QLabel(self.solution_tab)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(540, 200, 49, 16))
        sizePolicy.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy)
        self.label_28 = QLabel(self.solution_tab)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(390, 240, 21, 16))
        sizePolicy.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy)
        self.label_29 = QLabel(self.solution_tab)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(390, 220, 21, 16))
        sizePolicy.setHeightForWidth(self.label_29.sizePolicy().hasHeightForWidth())
        self.label_29.setSizePolicy(sizePolicy)
        self.label_30 = QLabel(self.solution_tab)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(540, 220, 49, 16))
        sizePolicy.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy)
        self.label_31 = QLabel(self.solution_tab)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(540, 260, 49, 16))
        sizePolicy.setHeightForWidth(self.label_31.sizePolicy().hasHeightForWidth())
        self.label_31.setSizePolicy(sizePolicy)
        self.wXText = QLineEdit(self.solution_tab)
        self.wXText.setObjectName(u"wXText")
        self.wXText.setGeometry(QRect(420, 180, 113, 21))
        sizePolicy2.setHeightForWidth(self.wXText.sizePolicy().hasHeightForWidth())
        self.wXText.setSizePolicy(sizePolicy2)
        self.label_32 = QLabel(self.solution_tab)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(540, 240, 49, 16))
        sizePolicy.setHeightForWidth(self.label_32.sizePolicy().hasHeightForWidth())
        self.label_32.setSizePolicy(sizePolicy)
        self.label_33 = QLabel(self.solution_tab)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(540, 140, 49, 16))
        sizePolicy.setHeightForWidth(self.label_33.sizePolicy().hasHeightForWidth())
        self.label_33.setSizePolicy(sizePolicy)
        self.iYText = QLineEdit(self.solution_tab)
        self.iYText.setObjectName(u"iYText")
        self.iYText.setGeometry(QRect(420, 240, 113, 21))
        sizePolicy2.setHeightForWidth(self.iYText.sizePolicy().hasHeightForWidth())
        self.iYText.setSizePolicy(sizePolicy2)
        self.label_34 = QLabel(self.solution_tab)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(390, 180, 21, 16))
        sizePolicy.setHeightForWidth(self.label_34.sizePolicy().hasHeightForWidth())
        self.label_34.setSizePolicy(sizePolicy)
        self.label_35 = QLabel(self.solution_tab)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(390, 160, 21, 16))
        sizePolicy.setHeightForWidth(self.label_35.sizePolicy().hasHeightForWidth())
        self.label_35.setSizePolicy(sizePolicy)
        self.wYText = QLineEdit(self.solution_tab)
        self.wYText.setObjectName(u"wYText")
        self.wYText.setGeometry(QRect(420, 200, 113, 21))
        sizePolicy2.setHeightForWidth(self.wYText.sizePolicy().hasHeightForWidth())
        self.wYText.setSizePolicy(sizePolicy2)
        self.label_36 = QLabel(self.solution_tab)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setGeometry(QRect(540, 160, 49, 16))
        sizePolicy.setHeightForWidth(self.label_36.sizePolicy().hasHeightForWidth())
        self.label_36.setSizePolicy(sizePolicy)
        self.label_37 = QLabel(self.solution_tab)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setGeometry(QRect(390, 280, 21, 16))
        sizePolicy.setHeightForWidth(self.label_37.sizePolicy().hasHeightForWidth())
        self.label_37.setSizePolicy(sizePolicy)
        self.label_38 = QLabel(self.solution_tab)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setGeometry(QRect(540, 280, 49, 16))
        sizePolicy.setHeightForWidth(self.label_38.sizePolicy().hasHeightForWidth())
        self.label_38.setSizePolicy(sizePolicy)
        self.bText = QLineEdit(self.solution_tab)
        self.bText.setObjectName(u"bText")
        self.bText.setGeometry(QRect(420, 280, 113, 21))
        sizePolicy2.setHeightForWidth(self.bText.sizePolicy().hasHeightForWidth())
        self.bText.setSizePolicy(sizePolicy2)
        self.tabWidget.addTab(self.solution_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow",
                                                             u"\u0412\u044b\u0447\u0438\u0441\u043b\u0438\u0442\u0435\u043b\u044c \u0433\u0435\u043e\u043c\u0435\u0442\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u0438\u0441\u0442\u0438\u043a",
                                                             None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"R<sub>\u0432\u043d</sub> =", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"L<sub>\u0441\u0442<sub/>", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"I<sub>\u043e\u0442</sub> =", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"t =", None))
        self.solveBtn.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0435\u0448\u0438\u0442\u044c", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.inputTab),
                                  QCoreApplication.translate("MainWindow", u"\u0412\u0432\u043e\u0434", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"R<sub>\u043e\u0441 =", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>2</sup>", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"A =", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>3</sup>", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"S<sub>x</sub> =", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>3</sup>", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"S<sub>y</sub> =", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"x<sub>c</sub> =", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"y<sub>c</sub> =", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>4</sup>", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"J<sub>x</sub> =", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>3</sup>", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"W<sub>x</sub> =", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"H =", None))
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"J<sub>y</sub> =", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>3</sup>", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"i<sub>y</sub> =", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"i<sub>x</sub> =", None))
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>4</sup>", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"W<sub>x</sub> =", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"J<sub>p</sub> =", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"\u0441\u043c<sup>4</sup>", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"B =", None))
        self.label_38.setText(QCoreApplication.translate("MainWindow", u"\u043c\u043c", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.solution_tab), QCoreApplication.translate("MainWindow",
                                                                                                        u"\u0420\u0435\u0448\u0435\u043d\u0438\u0435",
                                                                                                        None))
    # retranslateUi
