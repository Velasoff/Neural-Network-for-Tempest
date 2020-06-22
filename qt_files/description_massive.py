# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'description_massive.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(434, 420)
        self.desc_massive = QtWidgets.QTextEdit(Dialog)
        self.desc_massive.setGeometry(QtCore.QRect(0, 0, 451, 411))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        self.desc_massive.setPalette(palette)
        self.desc_massive.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.desc_massive.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.desc_massive.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.desc_massive.setTabChangesFocus(False)
        self.desc_massive.setUndoRedoEnabled(True)
        self.desc_massive.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.desc_massive.setReadOnly(True)
        self.desc_massive.setObjectName("desc_massive")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Свойства Выбранного массива"))
        self.desc_massive.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))

