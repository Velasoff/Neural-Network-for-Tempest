import math

from sklearn.metrics import mean_squared_error
import scipy.signal as signal
import numpy as np
from PyQt5 import QtCore
from sklearn.neural_network import MLPRegressor
import pathlib
from miscellaneous import interpolate_core

from a_tempest import train, mass, predict


class thread_mlp(QtCore.QThread):
    model = QtCore.pyqtSignal(MLPRegressor)
    exception = QtCore.pyqtSignal(str)

    def __init__(self, layers, iteration, solver, func, selection, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.layers = layers
        self.iteration = iteration
        self.solver = solver
        self.func = func
        self.selection = selection

    def run(self):
        try:
            model = train(self.layers, self.iteration, self.solver, self.func, self.selection)
            if math.isnan(model.loss_):
                raise Exception("Нейронная сеть не обучена. Необходимо увеличение количества итераций.")
            self.model.emit(model)
        except Exception as e:
            self.exception.emit(str(e))


class create_massive(QtCore.QThread):
    name_mass = QtCore.pyqtSignal(str)
    def __init__(self, m_min, m_max, m_num, m_dict, m_name, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.min = m_min
        self.max = m_max
        self.num = m_num
        self.dict = m_dict
        self.name = m_name

    def run(self):
        massive = mass(self.min, self.max, self.num, self.dict)
        pathlib.Path('./artificial_massive/').joinpath(pathlib.Path('./' + self.name + '/')).mkdir(exist_ok=True)
        with open(f'artificial_massive/{self.name}/desc.txt', 'wt', encoding='windows-1251') as f:
            f.write('Минимальное отношение сигнал/шум: ' + str(self.min) +
                    '\nМаксимальное отношение сигнал/шум: ' + str(self.max) + '\nКоличество выборок: ' +
                    str(self.num) + '\nПараметры выборки: ' + str(self.dict))
        np.save(f'artificial_massive/{self.name}/massive.npy', massive, allow_pickle=True)
        np.savez(f'artificial_massive/{self.name}/desc', min_SN=self.min,
                 max_SN=self.max, iteration=self.num, param=self.dict, allow_pickle=True)
        self.name_mass.emit(self.name)

class select_massive(QtCore.QThread):
    massive = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, name_massive, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.name = name_massive
        self.parent = parent

    def run(self):
        file = np.load(f'artificial_massive/{self.name}/desc.npz',
                            allow_pickle=True)
        massive = np.load(f'artificial_massive/{self.name}/massive.npy',
                               allow_pickle=True)

        self.massive.emit(massive)

class get_efficiency(QtCore.QThread):
    efficiency = QtCore.pyqtSignal(np.ndarray)
    progress = QtCore.pyqtSignal(int)
    def __init__(self, massive, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.massive = massive
        self.running = False

    def run(self):
        n = self.parent.iter
        clear = np.load('./data_for_train/trn_1.npy')
        clear = (clear - clear.min()) / (clear.max() - clear.min())
        r = np.empty(n)
        self.running = True
        for j in range(n):
            y_prepared = np.empty(self.parent.size_input)
            for i in range(self.parent.size_input[0]):
                y_prepared[i] = self.massive[j][i:i + self.parent.size_input[1]]
            filtered = predict(self.parent, y_prepared)
            filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
            r[j] = mean_squared_error(clear, filtered)
            self.progress.emit(j + 1)
        r = interpolate_core(r, n // 50)
        self.efficiency.emit(r)