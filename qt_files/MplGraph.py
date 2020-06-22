# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MplGraph.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from qt_files.mplwidget import MplWidget


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(790, 500)
        Dialog.setMinimumSize(QtCore.QSize(790, 500))
        Dialog.setMaximumSize(QtCore.QSize(790, 500))
        self.MplWidget = MplWidget(Dialog)
        self.MplWidget.setGeometry(QtCore.QRect(10, 50, 771, 441))
        self.MplWidget.setObjectName("MplWidget")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
