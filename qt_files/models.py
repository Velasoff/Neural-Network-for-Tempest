# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'models.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Models(object):
    def setupUi(self, Models):
        Models.setObjectName("Models")
        Models.setWindowModality(QtCore.Qt.WindowModal)
        Models.resize(320, 240)
        Models.setMinimumSize(QtCore.QSize(320, 240))
        Models.setMaximumSize(QtCore.QSize(320, 240))
        self.list_models = QtWidgets.QListWidget(Models)
        self.list_models.setGeometry(QtCore.QRect(10, 10, 301, 181))
        self.list_models.setMinimumSize(QtCore.QSize(301, 181))
        self.list_models.setMaximumSize(QtCore.QSize(301, 181))
        self.list_models.setFrameShape(QtWidgets.QFrame.Box)
        self.list_models.setObjectName("list_models")
        self.open = QtWidgets.QPushButton(Models)
        self.open.setEnabled(False)
        self.open.setGeometry(QtCore.QRect(60, 200, 75, 23))
        self.open.setObjectName("open")
        self.delete = QtWidgets.QPushButton(Models)
        self.delete.setEnabled(False)
        self.delete.setGeometry(QtCore.QRect(180, 200, 75, 23))
        self.delete.setObjectName("delete")

        self.retranslateUi(Models)
        QtCore.QMetaObject.connectSlotsByName(Models)

    def retranslateUi(self, Models):
        _translate = QtCore.QCoreApplication.translate
        Models.setWindowTitle(_translate("Models", "Выбрать модель из существующих"))
        self.open.setText(_translate("Models", "Открыть"))
        self.delete.setText(_translate("Models", "Удалить"))

