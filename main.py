import sys
from datetime import datetime

from PyQt5 import QtWidgets, QtGui

from MplAbstract import matplotlib_for_qt
from a_tempest import *
from miscellaneous import get_text_desc_mass
from n_tempest import tempest
from qt_files.Tempest import Ui_MainWindow
from qt_files.description_massive import Ui_Dialog
from qt_files.massive import Ui_Massive
from qt_files.models import Ui_Models
from threads import *

# Создание переменных для работы с файловой системой
data_path = pathlib.Path('./data/')
massive_path = pathlib.Path('./artificial_massive/')
picture_path = pathlib.Path('./picture/')
models_path = pathlib.Path('./neural_models/')
train_path = pathlib.Path('./data_for_train/')
# Функция, возвращающая реальное время
now = lambda: datetime.now()


# Класс создания окна загрузки моделей нейронной сети
class models_window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = Ui_Models()
        self.ui.setupUi(self)
        # Проверка существующих файлов в директории /neural_models/
        for i in models_path.iterdir():
            # Добавление существующих файлов в список моделей
            self.ui.list_models.addItem(str(i)[14:])

        # Соединение нажатия на элемент списка с функцией click()
        self.ui.list_models.clicked.connect(self.click)
        # Соединение двойного нажатия на элемент нажатия на кнопку Open с функцией open_model()
        self.ui.list_models.doubleClicked.connect(self.open_model)
        self.ui.open.clicked.connect(self.open_model)
        # Соединение двойного нажатия на элемент нажатия на кнопку Delete с функцией delete_model()
        self.ui.delete.clicked.connect(self.delete_model)

    # Функция срабатывает на нажатие на определенный элемент из списка и производит разблокировку кнопок в окне
    def click(self):
        self.ui.open.setEnabled(True)
        self.ui.delete.setEnabled(True)

    # Функция срабатывает при двойном нажатии на элемент или при нажатии на кнопку Open  выбранным элементом
    def open_model(self):
        # Записываем в переменную название выбранной модели
        self.txt = self.ui.list_models.currentItem().text()
        # Загружаем в переменную главного окна модель из файла
        self.parent.model = np.load(f'neural_models/' + self.txt + '/model.npy', allow_pickle=True).item()
        # Загружаем в переменную главного окна размерность
        # Фактически, можно избавится от этой переменной использованием атрибута hidden_layer из модели
        self.parent.size_input = np.load(f'neural_models/' + self.txt + '/size.npy')
        # Внесение изменений в интерфейс главного окна:
        # Ввод названия модели
        self.parent.ui.name_NN.setText(self.txt)
        # Активация индикатора загрузки модели
        self.parent.ui.NNready.setChecked(True)
        self.close()
        # Внесение сообщения об успешности в Логи
        self.parent.set_text_mlp('Нейронная сеть загружена.', typetxt="Excellent")
        # Установка имени модели в качестве основной в главном окне
        self.parent.name_mlp = self.txt

    # Функция срабатывает при нажатии на кнопку Delete
    def delete_model(self):
        # Удаление файлов модели и соответствующей папки
        models_path.joinpath(pathlib.Path('./' + self.txt + '/model.npy')).unlink()
        models_path.joinpath(pathlib.Path('./' + self.txt + '/size.npy')).unlink()
        models_path.joinpath(pathlib.Path('./' + self.ui.list_models.currentItem().text())).rmdir()
        # Перезаполнение списка моделей
        self.ui.list_models.clear()
        for i in models_path.iterdir():
            self.ui.list_models.addItem(str(i)[14:])
        # Блокировка кнопок, чтобы не возникло ошибки, когда модель из списка не выбрана, а кнопки нажимаются
        self.ui.open.setEnabled(False)
        self.ui.delete.setEnabled(False)


# Простейший класс создания окна описания массива
class description_massive(QtWidgets.QDialog):
    def __init__(self, parent=None, name_item=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModal)
        # Загрузка описания массива из файла и его размещение в окне с помощью создания HTML ода функцией setHtml()
        file = np.load(f'artificial_massive/{name_item}/desc.npz', allow_pickle=True)
        self.ui.desc_massive.setHtml(get_text_desc_mass(file['min_SN'].item(), file['max_SN'].item(),
                                                        file['iteration'].item(), file['param'].item()))


# Класс создания окна для определения параметров нового массива
class massive_window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Massive()
        self.ui.setupUi(self)
        self.parent = parent
        self.setWindowModality(QtCore.Qt.WindowModal)

        # При введении названия разблокируется кнопка начала создания массива
        self.ui.name_mass.textChanged.connect(self.enable)
        # При нажатии на кнопку ОК начинается создание массива
        self.ui.buttonBox.accepted.connect(self.create_massive)

    # Функция отвечающая за блокировку и разблокировку кнопки ОК
    def enable(self):
        if self.ui.name_mass.text() == '':
            self.ui.buttonBox.setEnabled(False)
        else:
            self.ui.buttonBox.setEnabled(True)

    # Функция, инициирующая создание массива
    def create_massive(self):
        # Создаётся список, который после цикла будет содержать кортежи с амплитудами и частотами гармонических шумов
        list_amp = []
        for i in range(self.ui.list_sin.rowCount()):
            list_amp.append(
                (np.float(self.ui.list_sin.item(i, 0).text()) / 100, np.float(self.ui.list_sin.item(i, 1).text())))
        # Словарь с параметрами будующего массива
        mass_dict = {'gaus': self.ui.SN_gaus.value() / 100,
                     'impuls': self.ui.SN_impuls.value() / 100, 'num_impuls': self.ui.num_impuls.value(),
                     'radio_impuls': self.ui.SN_radio.value() / 100, 'num_radio': self.ui.num_radio.value(),
                     'amp_freq': list_amp
                     }
        # Закрытие окна и передача параметров в функцию new_tempest() главного окна
        self.close()
        self.parent.new_tempest(self.ui.min_SN.value(), self.ui.max_SN.value(), self.ui.num_mass.value(), mass_dict,
                                self.ui.name_mass.text())


# Класс создания главного окна
class Tempest(QtWidgets.QMainWindow):
    def __init__(self):
        super(Tempest, self).__init__()
        self.massive_window = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Список функций и сигналов, обработка которых происходит в слое MLP
        # Строка с функцией активации
        self.func = 'identity'
        # Строка с методом обучения нейронной сети
        self.solver = 'sgd'
        # Строка с именем нейронной сети; эта и две прошлых строки всего лишь инициирующие переменные,
        # которые потребуются позже
        self.name_mlp = None
        # Загрузка примера входного и целевого сигналов
        self.target_signal = np.load('./data_for_train/trn_1.npy')
        self.input_signal = np.load('./data_for_train/sgnl_1.npy')
        # При изменении количества слоёв вызывается функция change_layer_mlp()
        self.ui.layer_preset.currentIndexChanged.connect(self.change_layer_mlp)
        # При изменении функции активации вызывается set_func_mlp()
        self.ui.function.currentIndexChanged.connect(self.set_func_mlp)
        # При изменении метода обучения вызывается set_solver_mlp()
        self.ui.solver.currentIndexChanged.connect(self.set_solver_mlp)
        # При нажатии кнопки Обучить вызывается preset_mlp()
        self.ui.button_learning.clicked.connect(self.preset_mlp)
        # При нажатии на Загрузить или сохранить вызываются load_mlp() и save_mlp() соответственно
        self.ui.load_NN.clicked.connect(self.load_mlp)
        self.ui.save_NN.clicked.connect(self.save_mlp)
        # Инициируется размерность
        self.size_input = 0
        # Заполняется список обучающих выборок из файлов desc_?.txt
        for i in train_path.iterdir():
            if i.name.startswith('desc_'):
                with open(str(i), 'rt', encoding='utf-8') as f:
                    d = f.read().replace('\ufeff', '')
                    self.ui.selection.addItem(str(d))

        # Список функций и сигналов, обработка которых происходит в Synth
        # При нажатии на элемент из списка вызывается mass_click_synth()
        self.ui.list_massive.clicked.connect(self.mass_click_synth)
        # При нажатии на Закрепить вызывается massive_selected_synth()
        self.ui.select_mass.clicked.connect(self.massive_selected_synth)
        # При нажатии на кнопку Добавить вызывается функция open_mass_synth()
        self.ui.add_mass.clicked.connect(self.open_mass_synth)
        # При нажатии на Удалить вызывается delete_mass_synth()
        self.ui.delete_mass.clicked.connect(self.delete_mass_synth)
        # При двойном нажатии на элемент из списка вызывается open_desc_synth()
        self.ui.list_massive.doubleClicked.connect(self.open_desc_synth)
        # При нажатии на кнопку Построить график вызывается plot_massive_synth()
        self.ui.plot_mass.clicked.connect(self.plot_massive_synth)
        # При нажатии на Пересчитать... вызывается get_efficiency_synth()
        self.ui.get_efficiency.clicked.connect(self.get_efficiency_synth)
        # При нажатии на Построить вызывается plot_efficiency_synth()
        self.ui.plot_efficiency.clicked.connect(self.plot_efficiency_synth)
        # При выборе моделей нейронных сетей из списка вызывается set_NN_for_massive()
        self.ui.NN_for_massive.currentIndexChanged.connect(self.set_NN_for_massive)
        # Заполнятеся список доступных массивов
        for i in massive_path.iterdir():
            self.ui.list_massive.addItem(str(i)[19:])

        # Список функций и сигналов, обработка которых происходит в слое Tempest
        # Заполнение списка имеющихся реальных сигналов
        for i in data_path.iterdir():
            self.ui.list_tempest.addItem(str(i)[5:])
        # При нажатии на элемент из списка вызывается click_tempest()
        self.ui.list_tempest.clicked.connect(self.click_tempest)
        # При двойном нажатии на элемент из списка и при нажатии кнопки Закрепить вызывается select_tempest()
        self.ui.list_tempest.doubleClicked.connect(self.select_tempest)
        self.ui.select_tempest.clicked.connect(self.select_tempest)
        # При нажатии на кнопку Построить график вызывается plotting_tempest()
        self.ui.plot_tempest.clicked.connect(self.plotting_tempest)
        # При нажатии на кнопку Обработать... вызывается processing_tempest()
        self.ui.process_nn.clicked.connect(self.processing_tempest)
        # При нажатии на кнопку Построить обработанный сигнал вызывается plot_processed_tempest()
        self.ui.plot_processing.clicked.connect(self.plot_processed_tempest)

    # Следующий блок функций относится к блоку MLP
    # Функция установки определенного метода обучения при изменении элемента из списка
    def set_solver_mlp(self):
        if self.ui.solver.currentIndex() == 0:
            self.solver = 'sgd'
        elif self.ui.solver.currentIndex() == 1:
            self.solver = 'lbfgs'
        elif self.ui.solver.currentIndex() == 2:
            self.solver = 'adam'

    # Функция установки функции активации при изменении элемента из списка
    def set_func_mlp(self):
        if self.ui.function.currentIndex() == 0:
            self.func = 'identity'
        elif self.ui.function.currentIndex() == 1:
            self.func = 'relu'
        elif self.ui.function.currentIndex() == 2:
            self.func = 'logistic'
        elif self.ui.function.currentIndex() == 3:
            self.func = 'tanh'

    # Изменение блокировок счетчиков при изменении количества слоёв в списке
    def change_layer_mlp(self):
        if self.ui.layer_preset.currentIndex() == 0:
            self.ui.third_layer.setEnabled(False)
            self.ui.fourth_layer.setEnabled(False)
        elif self.ui.layer_preset.currentIndex() == 1:
            self.ui.third_layer.setEnabled(True)
            self.ui.fourth_layer.setEnabled(False)
        elif self.ui.layer_preset.currentIndex() == 2:
            self.ui.third_layer.setEnabled(True)
            self.ui.fourth_layer.setEnabled(True)

    # Функция, инициирующая создание нейронной сети
    def preset_mlp(self):
        # Сброс галочки готовности нейронной сети
        self.ui.NNready.setChecked(False)
        # Создание лога
        self.set_text_mlp(f'Обучение нейронной сети началось.')
        # Блокировка кнопок во избежании эксцессов
        self.ui.load_NN.setEnabled(False)
        self.ui.save_NN.setEnabled(False)
        self.ui.button_learning.setEnabled(False)
        # Установка размерности
        self.size_input = [32000 - self.ui.first_layer.value(), self.ui.first_layer.value()]
        # Создание списка из количества нейронов в каждом слое
        if self.ui.layer_preset.currentIndex() == 0:
            self.layers = [self.ui.first_layer.value(), self.ui.second_layer.value()]
        elif self.ui.layer_preset.currentIndex() == 1:
            self.layers = [self.ui.first_layer.value(), self.ui.second_layer.value(), self.ui.third_layer.value()]
        elif self.ui.layer_preset.currentIndex() == 2:
            self.layers = [self.ui.first_layer.value(), self.ui.second_layer.value(),
                           self.ui.third_layer.value(), self.ui.fourth_layer.value()]
        # Создание и запуск потока, который обучает нейронную сеть
        self.thread_mlp = thread_mlp(self.layers, self.ui.iteration.value(), self.solver, self.func,
                                     self.ui.selection.currentIndex())
        self.thread_mlp.model.connect(self.well_mlp, QtCore.Qt.QueuedConnection)
        self.thread_mlp.exception.connect(self.fail_mlp, QtCore.Qt.QueuedConnection)
        self.thread_mlp.start()

    # Функция, вызывающаяся после успешного создания модели
    def well_mlp(self, model):
        # Запись модели в соответствующую переменную
        self.model = model
        # установка галочки готовности НС
        self.ui.NNready.setChecked(True)
        # Запись в лог
        self.set_text_mlp(
            f'Обучение нейронной сети закончилось успешно!',
            typetxt='Excellent')
        # Отображение названия нейронной сети
        self.ui.name_NN.setText(self.get_name_mlp())
        # Разблокировка кнопок
        self.ui.load_NN.setEnabled(True)
        self.ui.save_NN.setEnabled(True)
        self.ui.button_learning.setEnabled(True)
        # Сброс готового имени нейронной сети
        self.name_mlp = None
        # Разблокировка кнопок, связанных с готовностью нейронной сети
        if self.ui.massive_2.isChecked():
            self.ui.process_nn.setEnabled(True)
        if self.ui.massive.isChecked():
            self.ui.get_efficiency.setEnabled(True)

    # Функция, вызывающаяся при неудачной попытке создания нейронной сети записывает в лог ошибку
    def fail_mlp(self, exception):
        self.set_text_mlp(
            'Ошибка: ' + exception,
            typetxt='Error')

    # Функция создания названия нейронной сети
    def get_name_mlp(self):
        # Если название не предустановлено, то оно создаётся из параметров существующей нейронной сети
        if self.name_mlp == None:
            str_mlp = ''
            str_mlp += 'Neural_model_of_'
            for i in range(len(self.layers)):
                str_mlp += str(self.layers[i])
                str_mlp += '_'
            str_mlp += self.func
            str_mlp += "_"
            str_mlp += self.solver
            str_mlp += "_"
            str_mlp += "sel"
            str_mlp += str(self.ui.selection.currentIndex())
            return str_mlp
        else:
            return self.name_mlp

    # Создание сообщения лога
    def set_text_mlp(self, txt, typetxt='Normal'):
        # Установка корректного реального времени
        if len(str(now().hour)) == 1:
            hour = '0' + str(now().hour)
        else:
            hour = str(now().hour)
        if len(str(now().minute)) == 1:
            minute = '0' + str(now().minute)
        else:
            minute = str(now().minute)
        if len(str(now().second)) == 1:
            second = '0' + str(now().second)
        else:
            second = str(now().second)
        # Запись времени выведения лога
        self.ui.log_NN.insertHtml(f'<a style="color: green; font-family: Arial">{hour}:{minute}:{second}: <a>')
        # Обычное сообщение
        if typetxt == 'Normal':
            self.ui.log_NN.insertHtml(f'<a>{txt}<br /><a>')
        # Ошибка
        elif typetxt == 'Error':
            self.ui.log_NN.insertHtml(f'<h3 style="color: red; font-family: Arial">{txt}<br /></h3>')
        # Удачное завершение
        elif typetxt == 'Excellent':
            self.ui.log_NN.insertHtml(f'<h3>{txt}<br /></h3>')

    # Функция сохранения обученной нейронной сети
    def save_mlp(self):
        models_path.joinpath(pathlib.Path('./' + self.get_name_mlp() + '/')).mkdir(exist_ok=True)
        np.save(f'neural_models/' + self.get_name_mlp() + '/model.npy', self.model, allow_pickle=True)
        np.save(f'neural_models/' + self.get_name_mlp() + '/size.npy', self.size_input, allow_pickle=True)

    # Функция загрузки нейронной сети
    def load_mlp(self):
        # Создается объект окна выведения моделей и отображается
        self.models_window = models_window(self)
        self.models_window.show()
        # Неправильный ход, но единственный
        # Производится разблокировка кнопок, связанных с готовностью нейронной сети
        # Ход неправильный, так как даже если окно выведения моделей будет закрыто без выбора модели
        # кнопки всё равно активируются
        if self.ui.massive_2.isChecked():
            self.ui.process_nn.setEnabled(True)
        if self.ui.massive.isChecked():
            self.ui.get_efficiency.setEnabled(True)

    # Следующий блок функций относится к вкладке Tempest
    # При нажатии на элемент из списка электромагнитных сигналов:
    def click_tempest(self):
        # Запись названия выбранного сигнала
        self.txt = self.ui.list_tempest.currentItem().text()
        # Устанавливается миниатюра в окошке
        pic = QtGui.QPixmap('picture/mini_tempest/' + self.txt[:-4] + '.jpg')
        self.ui.picture.setPixmap(pic)
        # Становится активной кнопка закрепления сигнала
        self.ui.select_tempest.setEnabled(True)

    # При закрепления сигнала как основного:
    def select_tempest(self):
        # Вывод названия сигнала
        self.ui.selected_tempest.setText(self.txt)
        # Отображение индикатора закрепления сигнала
        self.ui.massive_2.setChecked(True)
        # Здесь выступает непростой механизм. Так как нейронная сеть принимает на вход определенное количество отсчетов
        # а класс tempest содержит сигнал пригодный для обработки в нейронной сети, то необходимо вписывать размерность
        # к которой необходимо привести сигнал.
        if self.size_input is 0:
            self.tempest = tempest('./data/' + self.txt, self.ui.first_layer.value())
        else:
            self.tempest = tempest('./data/' + self.txt, self.size_input[1])
        # Установка описания сигнала, взятого из шапки файла *.DAT
        self.ui.tempest_txt.setText(str(self.tempest))
        # Активация кнопок
        self.ui.plot_tempest.setEnabled(True)
        self.ui.plot_processing.setEnabled(False)
        if self.ui.NNready.isChecked():
            self.ui.process_nn.setEnabled(True)

    # Метод для построения графика функции
    def plotting_tempest(self):
        self.plot_window = matplotlib_for_qt(self, self.tempest.signal, name=self.txt)
        self.plot_window.show()

    # Метод для обработки сигнала нейронной сетью
    def processing_tempest(self):
        self.tempest.adaptive_filter(self)
        self.processed_tempest = self.tempest.adapted_signal
        self.ui.plot_processing.setEnabled(True)

    # Метод для построения графика сигнала
    def plot_processed_tempest(self):
        self.plot_window = matplotlib_for_qt(self, self.processed_tempest,
                                             name=self.txt + ' Neural network\n' + self.ui.name_NN.text())
        self.plot_window.show()

    # Следующий блок функций относиться ко вкладке Synth
    # Метод открывает окно создания массива
    def open_mass_synth(self):
        self.massive_window = massive_window(self)
        self.massive_window.show()

    # Метод принимает данные о массиве, создает поток для создания массива и отправляет ему данные
    def new_tempest(self, mass_min, mass_max, mass_num, mass_dict, mass_name):
        self.thread_cmass = create_massive(mass_min, mass_max, mass_num, mass_dict, mass_name)
        # Соединение атрибута класса потока name_mass с функцией add_tempest
        # То есть, когда данная переменная изменяется, это вызывает функцию и может передавать в неё сигнал из потока
        self.thread_cmass.name_mass.connect(self.add_tempest, QtCore.Qt.QueuedConnection)
        self.thread_cmass.start()

    # Добавление имени массива в список
    def add_tempest(self, name):
        self.ui.list_massive.addItem(name)

    # Метод, реализующий удаление массива как из списка, так и с файловой системы
    def delete_mass_synth(self):
        # Фиксируется название массива
        name = self.ui.list_massive.currentItem().text()
        # Перебираются все файлы в папке и удаляются
        for i in massive_path.joinpath(pathlib.Path('./' + name + '/')).iterdir():
            i.unlink()
        # Удаляется сама папка
        massive_path.joinpath(pathlib.Path('./' + name + '/')).rmdir()
        # Список очищается и перезаполняется заново
        self.ui.list_massive.clear()
        for i in massive_path.iterdir():
            self.ui.list_massive.addItem(str(i)[19:])
        # Кнопки блокируются, чтобы не было ошибки, когда происходит нажатие, а элемент из списка не выбран
        self.ui.delete_mass.setEnabled(False)
        self.ui.select_mass.setEnabled(False)

    # Метод, открывающий окно описания массива
    def open_desc_synth(self):
        self.desc_window = description_massive(self, self.ui.list_massive.currentItem().text())
        self.desc_window.show()

    # Метод, активирующий кнопки при нажатии на элемент списка
    def mass_click_synth(self):
        self.ui.delete_mass.setEnabled(True)
        self.ui.select_mass.setEnabled(True)

    # Метод, создающий поток для загрузки выбранного массива в оперативную память
    # Так как массивы весят много, их загрузка происходит долго, поэтому необходим дополнительный поток
    def massive_selected_synth(self):
        # Создание объекта потока
        self.thread_smass = select_massive(self.ui.list_massive.currentItem().text(), self)
        # При запуке потока вызывается метод start_select_synth
        self.thread_smass.started.connect(self.start_select_synth)
        # При изменении атрибута massive в классе потока, вызывается метод finish_select_synth
        # с вводом в него данного сигнала
        self.thread_smass.massive.connect(self.finish_select_synth, QtCore.Qt.QueuedConnection)
        # Запуск потока
        self.thread_smass.start()

    # Метод, вызывающийся при запуске потока загрузки массива
    def start_select_synth(self):
        self.ui.NN_for_massive.setEnabled(False)
        self.ui.NN_for_massive.clear()
        self.ui.NN_for_massive.addItem('<Новая оценка>')
        self.ui.select_mass.setEnabled(False)
        self.ui.range_mass.setEnabled(False)
        self.ui.plot_mass.setEnabled(False)
        self.ui.get_efficiency.setEnabled(False)
        self.ui.plot_efficiency.setEnabled(False)
        self.ui.progressBar.setEnabled(False)
        self.ui.name_NN_2.clear()
        self.ui.progressBar.setValue(0)
        self.ui.massive.setChecked(False)

    def finish_select_synth(self, massive):
        with np.load(f'artificial_massive/{self.ui.list_massive.currentItem().text()}/desc.npz',
                     allow_pickle=True) as f:
            self.min = f["min_SN"].item()
            self.max = f["max_SN"].item()
            self.iter = f["iteration"].item()
        self.ui.range_mass.setMinimum(self.min)
        self.ui.range_mass.setMaximum(self.max)
        self.ui.range_mass.setSingleStep((self.max - self.min) / (self.iter - 1))
        self.ui.progressBar.setMaximum(self.iter)

        self.massive = massive
        self.ui.select_tempest.setEnabled(True)
        self.ui.massive.setChecked(True)
        self.ui.plot_mass.setEnabled(True)
        self.ui.range_mass.setEnabled(True)
        self.ui.selected_tempest_2.setText(self.ui.list_massive.currentItem().text())
        self.ui.plot_efficiency.setEnabled(False)
        if self.ui.NNready.isChecked():
            self.ui.get_efficiency.setEnabled(True)
        for i in massive_path.joinpath(pathlib.Path(f'./{self.ui.list_massive.currentItem().text()}/')).iterdir():
            if str(i.name)[:10] == 'efficiency':
                self.ui.NN_for_massive.setEnabled(True)
                self.ui.NN_for_massive.addItem(str(i.name)[11:-4])
        else:
            self.ui.NN_for_massive.setEnabled(True)

    def plot_massive_synth(self):
        self.plot_window = matplotlib_for_qt(self,
                                             self.massive[int((self.ui.range_mass.value() - self.min) * self.iter //
                                                              (self.max - self.min) - 1)],
                                             name=self.ui.list_massive.currentItem().text())
        self.plot_window.show()

    def set_NN_for_massive(self):
        if self.ui.NN_for_massive.currentIndex() == 0:
            self.ui.get_efficiency.setEnabled(True)
            self.ui.plot_efficiency.setEnabled(False)
            self.ui.name_NN_2.clear()
            self.ui.progressBar.setValue(0)
        else:
            self.ui.name_NN_2.setText(self.ui.NN_for_massive.currentText())
            try:
                self.score = np.load(f'artificial_massive/{self.ui.list_massive.currentItem().text()}'
                                     f'/efficiency_{self.ui.NN_for_massive.currentText()}.npy')
            except Exception:
                pass
            self.ui.get_efficiency.setEnabled(False)
            self.ui.plot_efficiency.setEnabled(True)
            self.name_mlp = self.ui.NN_for_massive.currentText()

    def get_efficiency_synth(self):
        for i in massive_path.joinpath(pathlib.Path(f'./{self.ui.list_massive.currentItem().text()}/')).iterdir():
            if str(i.name)[:10] == 'efficiency':
                if str(i.name)[11:-4] == self.ui.name_NN.text():
                    self.ui.name_NN_2.setText(self.ui.name_NN.text())
                    self.score = np.load(f'artificial_massive/{self.ui.selected_tempest_2.text()}'
                                         f'/efficiency_{self.ui.name_NN.text()}.npy')
                    self.ui.get_efficiency.setEnabled(False)
                    self.ui.plot_efficiency.setEnabled(True)
                    self.name_mlp = self.ui.name_NN.text()
                    return 0
        self.efficiency = get_efficiency(self.massive, self)
        self.efficiency.started.connect(self.start_efficiency_synth)
        self.efficiency.progress.connect(self.progress_synth, QtCore.Qt.QueuedConnection)
        self.efficiency.efficiency.connect(self.finish_efficiency_synth, QtCore.Qt.QueuedConnection)
        self.efficiency.start()

    def start_efficiency_synth(self):
        self.ui.select_mass.setEnabled(False)
        self.ui.range_mass.setEnabled(False)
        self.ui.plot_mass.setEnabled(False)
        # self.ui.get_efficiency.setText('Прервать')
        # self.ui.get_efficiency.clicked.connect(self.cancel_efficiency_synth)
        self.ui.plot_efficiency.setEnabled(False)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setEnabled(True)
        self.ui.get_efficiency.setEnabled(False)
        self.ui.name_NN_2.setText(self.get_name_mlp())
        self.ui.NN_for_massive.setEnabled(False)

    # def cancel_efficiency_synth(self):
    #     self.efficiency.running = False
    #     self.ui.get_efficiency.setText('Пересчитать на выбранном массиве')
    #     self.ui.get_efficiency.clicked.connect(self.get_efficiency_synth)
    #     self.ui.select_mass.setEnabled(True)
    #     self.ui.range_mass.setEnabled(True)
    #     self.ui.plot_mass.setEnabled(True)
    #     self.ui.progressBar.setEnabled(False)
    #     self.efficiency.terminate()

    def progress_synth(self, j):
        self.ui.progressBar.setValue(j)

    def finish_efficiency_synth(self, efficiency):
        self.score = efficiency
        np.save(f'artificial_massive/{self.ui.selected_tempest_2.text()}/efficiency_{self.ui.name_NN_2.text()}.npy',
                efficiency, allow_pickle=True)
        self.ui.select_mass.setEnabled(True)
        self.ui.range_mass.setEnabled(True)
        self.ui.plot_mass.setEnabled(True)
        self.ui.progressBar.setEnabled(False)
        self.ui.plot_efficiency.setEnabled(True)
        self.ui.NN_for_massive.setEnabled(True)
        self.ui.NN_for_massive.addItem(self.ui.name_NN_2.text())

    def plot_efficiency_synth(self):
        self.plot_window = matplotlib_for_qt(self, -self.score, scale=np.linspace(self.min, self.max, self.iter),
                                             name=self.ui.list_massive.currentItem().text() + ' efficiency\n' +
                                                  self.get_name_mlp())
        self.plot_window.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Tempest()
    window.show()
    sys.exit(app.exec_())
