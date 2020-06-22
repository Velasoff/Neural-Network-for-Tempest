from PyQt5.QtWidgets import QMainWindow
from qt_files.MplGraph import Ui_Dialog
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as toolbar_qt


class matplotlib_for_qt(QMainWindow):
    def __init__(self, parent=None, arr=[],  scale=None,  name='Matplotlib for QT'):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        if scale is None:
            scale = range(len(arr))


        self.setWindowTitle(name)
        self.addToolBar(toolbar_qt(self.ui.MplWidget.canvas, self))

        self.ui.MplWidget.canvas.axes.plot(scale, arr)
        # self.ui.MplWidget.canvas.axes.legend('arr')
        self.ui.MplWidget.canvas.axes.set_title(name)
        self.ui.MplWidget.canvas.draw()
