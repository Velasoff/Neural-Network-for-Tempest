from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from miscellaneous import *
import pathlib
train_path = pathlib.Path('./data_for_train/')


#     def __init__(self, input_nodes,
#                  neurons=[100, 10],
#                  iterations=1000,
#                  solver='sgd',
#                  activation='identity'):
#         # Вектор size содержит значения количества обучающих выборок и количество значений на одну выборку
#         self.size = [32000 - input_nodes, input_nodes]
#         # Две следующие операции производят импорт специально подготовленных обучающих массивов
#         self.target_signal = np.load('./data_for_train/trn.npy')
#         self.input_signal = np.load('./data_for_train/sgnl.npy')
#         # Создаются пустые массивы размерностью size
#         self.target_prepared = np.empty(self.size)
#         self.input_prepared = np.empty(self.size)
#         # Массивы заполняются посредством итерационного заполнения из подготовленных массивов
#         for i in range(self.size[0]):
#             self.target_prepared[i] = self.target_signal[i:i + input_nodes]
#             self.input_prepared[i] = self.input_signal[i:i + input_nodes]
#         # Внутренняя функция создаёт объект класса MLPRegressor
#         self.neural_model = self.__train(neurons, iterations, solver, activation)


def train(neurons, iterations, solver, activation, selection):
    # Создание обучающий выборок посредством случайного разбиения подготовленных массивов, с учетом того, что
    # будет задействовано test_size*100 процентов всего материала
    # MLP - модель нейронной сети
    #
    # Параметры:
    # ---------
    #
    #     random_state - предоставляет начальное значение для функций случайных чисел, позволяет синхронизировать
    #         функции пользующимися генератором случайных чисел
    #
    #     iterations - максимальное количество итераций; увеличение этого параметра ведет к увеличению времени
    #                 создания нейронной сети, но позволяет обучать более сложные модели
    #
    #     solver - решающий алгоритм для оптимизации процесса нахождения весовых коэффициентов
    #                 Может принимать следующие значения:
    #
    #                     - 'lbfgs' оптимизатор из семейства квази-Ньютоновских методов
    #
    #                     - 'sgd' Стохастический градиентный спуск
    #
    #                     - 'adam' метод, основанный на 'sgd' и созданный Kingma, Diederik, и Jimmy Ba
    #                         Эфективен только на большом количестве обучающих выборок (1000 и более)
    #
    #      neurons - вектор, задающий структуру нейронной сети, так размер этого вектора равен количеству слоев,
    #                 и каждое значение указывает на количество нейронов в определенном слое
    #
    #      activation - функция активации каждого нейрона
    #                 Может принимать значения:
    #
    #                     - 'identity', линейная функция,
    #                         возвращает f(x) = x
    #
    #                     - 'logistic', сигмоидальная функция,
    #                         возвращает f(x) = 1 / (1 + exp(-x)).
    #
    #                     - 'tanh', функция гиперболического тангенса,
    #                         возвращает f(x) = tanh(x).
    #
    #                     - 'relu', положительная линейная функция,
    #                         возвращает f(x) = max(0, x)
    #
    # Функция fit() позволяет обучать модель на основе обучающих выборок

    MLP = MLPRegressor(max_iter=iterations,
                       solver=solver,
                       random_state=42,
                       hidden_layer_sizes=neurons,
                       activation=activation,
                       learning_rate='adaptive')
    if selection == 0:
        tmp = 0
        for i in train_path.iterdir():
            if i.name.startswith('trn'):
                if int(i.name[-5]) > tmp:
                    tmp = int(i.name[-5])
        for i in range(1, tmp + 1):
            size = [4 * (32000 - neurons[0]), neurons[0]]
            sgnl = np.load(f'data_for_train/sgnl_{i}.npy')
            trn = np.load(f'data_for_train/trn_{i}.npy')
            target_prepared = np.empty(size)
            input_prepared = np.empty(size)
            for j in range(size[0] // 4):
                target_prepared[j] = sgnl[j:j + size[1]]
                input_prepared[j] = trn[j:j + size[1]]
    else:
        size = [(32000 - neurons[0]), neurons[0]]
        sgnl = np.load(f'data_for_train/sgnl_{selection}.npy')
        trn = np.load(f'data_for_train/trn_{selection}.npy')
        target_prepared = np.empty(size)
        input_prepared = np.empty(size)
        for j in range(size[0]):
            target_prepared[j] = sgnl[j:j + size[1]]
            input_prepared[j] = trn[j:j + size[1]]
    X_train, X_test, y_train, y_test = train_test_split(target_prepared, input_prepared, random_state=42, test_size=0.8)
    MLP = MLP.fit(X_train, y_train)
    return MLP


# Функция predict позволяет обрабатывать необходимые массивы данных, подобные начальной, обрабатывая
# полученный моделью
def predict(window, y):
    y_result = np.zeros(y.shape[0] + y.shape[1])
    result = window.model.predict(y)
    for i in range(y.shape[0]):
        y_result[i:i + y.shape[1]] += result[i]
    return y_result


# Функция mass позволяет составлять настраиваемые массивы, симулирующих электромагнитные волны.
# Настройка подразумевает контролировать составляющие электромагнитного шума и прибавлять к реальному
# чистому массиву, содержащему сигналы с клавиатуры компьютера.
# В массиве будут содержаться определенное количество выборок с разными значениями отношения сигнал/шум,
# лежащими в определенном диапазоне.
#
#     Параметры:
#     ----------
#
#         start_NS - минимальное значение отношения сигнал/шум
#
#         end_NS - максимальное значение отношения сигнал/шум
#
#         params - словарь, содержащий общие параметры требующегося массива
#             Значения (взяты из функции make_some_noise):
#                 gaus - амплитуда гауссово шума
#                 impuls - амплитуда импульсного шума
#                 num_impuls - количество импульсных возбуждений на всю выборку
#                 radio_impuls и num_radio - аналогичные параметры только для радио-импульсов
#                 amp_freq - словарь, содержащий кортежи из амплитуды и частоты гармоник синусоидального сигнала
#                 amp - общая величина амплитуды шума

def mass(start_NS: float, end_NS: float, n: int, params: dict):
    # Чистый сигнал, для последующего заполнения его шумами
    clear = np.load('./data_for_train/trn_1.npy')
    # Создаем и заполняем массив сигналов с разными коэффициентами шум/сигнал
    mass = np.empty((n, 32000))
    for i, item in enumerate(np.logspace(start_NS / 20, end_NS / 20, n)):
        mass[i] = clear + make_some_noise(gaus=params["gaus"],
                                          impuls=params["impuls"], num_impuls=params["num_impuls"],
                                          radio_impuls=params["radio_impuls"],
                                          num_radio=params["num_radio"],
                                          amp_freq=params["amp_freq"],
                                          amp=clear.max() * (1 / item))
    # Приводим к отношению сигнал/шум, и создаем соответствующий массив значений
    S_N = np.logspace(start_NS / 20, end_NS / 20, n)
    return mass
