from scipy import stats
from scipy.fftpack import fft, ifft
from miscellaneous import *
from a_tempest import predict


# Класс Tempest принимает на вход файлы .DAT, обрабатывает их и производит манипуляции через встроенные методы
class tempest(object):
    def __init__(self, a, size=100):
        # Открытие файла формата *.DAT
        with open(a, 'rt', encoding='windows-1251') as f:
            # Так как все значения в файле разделены точкой с запятой, а также существуют отдельные строки, значит
            # сначала избавляемся от строк, путем удаления символов переноса, а потом разделяем значения и полуается
            # список. Стоит рассмотреть другой вариант: разделение по строкам, а потом уже по элементам строки,
            # тогда получится матрица, которую легче будет обрабатывать
            d = f.read().replace('\n', '').split(';')
            # Записываем название файла в отдельный атрибут
            self.name = f.name
        # В документации было поставлено условие, что последный элемент в шапке заканчивается строкой Values
        # В данной части находится этот элемент и выделяется в отдельный атрибут
        for num, item in enumerate(d):
            if item == 'Values':
                self.start_point = num
                break
        # Сигнал
        self.signal = np.float_(d[self.start_point+3::2]) - np.average(np.float_(d[self.start_point+3::2]))
        # Количество отсчётов во всем сигнале
        self.max_points = int(d[self.start_point+1]) - 1
        # Шапка файла .DAT, содержащая основную информацию по сигналу
        self.head = d[0:self.start_point+2]
        # Временной массив
        self.time = np.arange(len(self.signal)) * np.float_(d[self.start_point+202]).round(6) / 100
        # Преобразование Фурье
        self.__fourier()
        # Обратное преобразование Фурье (выдает первоначальны сигнал).
        # Добавлена на случай, если функция plot() вызвана раньше каких-либо действий со спектром
        self.__fourier(False)
        # Так как действия со спектром происходят с переменной fourier, то следует следить за её изменением для
        # избежания повторного её использования
        self.reused = False
        # Frequency sampling
        self.fs = int(1 / (self.time[1] - self.time[0]))
        # Матрица для ввода в адаптивный фильтр
        self.prepared_signal = np.empty([self.max_points - size, size])
        # Размерность необходима для определения формы массива, которы потом будет вводится в адаптивный фильтр
        self.size = size
        self.__transform_adaptive(mode='to')
        # adapted_signal - отфильтрованный сигнал
        self.adapted_signal = np.zeros(self.signal.shape)
        # нормализованный сигнал
        self.normalized_signal = self.signal.copy()

    # Производит прямое (если b=True, по умолчанию) и обратное (если b=False) преобразование Фурье
    def __fourier(self, b=True):
        if b:
            self.fourier = fft(self.signal)
        else:
            self.post_signal = np.real(ifft(self.fourier))

    # Функция построения графика объекта
    def plot(self):
        plt.figure(figsize=(8, 8), dpi=80)
        plt.subplot(3, 1, 1)
        plt.plot(self.time, self.signal, 'b')
        plt.title(self.name)
        plt.ylabel('Amp ' + self.head[27])
        plt.xlabel('Time ' + self.head[50])

        plt.subplot(3, 1, 2)
        plt.plot(np.linspace(0, self.fs, 32000), np.abs(self.fourier), 'r')
        plt.xlabel('Freq Hz')
        plt.ylabel('Amp')

        plt.subplot(3, 1, 3)
        plt.plot(self.time, self.post_signal, 'g')
        plt.ylabel('Amp ' + self.head[27])
        plt.xlabel('Time ' + self.head[50])

        plt.show()

    # Функция построения и применения БИХ-фильтра
    def iir(self, wp, ws, gpass, gstop):
        b, a = signal.iirdesign(wp, ws, gpass, gstop, ftype='cheby2')
        _, self.h = signal.freqs(b, a, worN=16000)
        self.h = np.concatenate((self.h, self.h[::-1]), axis=None)
        self.post_signal = signal.lfilter(b, a, self.signal)

        # b, a = signal.iirdesign(0.1, 0.2, 2, 20, ftype='cheby2')
        # _, h = signal.freqs(b, a, worN=16000)
        # h = np.concatenate((h, h[::-1]), axis=None)
        # self.h *= h
        # self.post_signal = signal.lfilter(b, a, self.post_signal)

    # Награда для обучения с подкреплением
    def reward(self):
        return stats.sem(self.adapted_signal)

    # Сделать Гауссову оконную функцию
    def fir(self, edge):
        if self.reused:
            self.__fourier()
            self.reused = False
        # edges = [0, 0.45, 0.48, 0.52, 0.55, 1]
        # taps = signal.remez(10, edges, [0, 1, 0], Hz=2)
        # w, self.h = signal.freqz(taps, [1], worN=16000)
        b, a = signal.ellip(4, 5, 20, edge, 'bandpass', analog=True)
        w, self.h = signal.freqs(b, a, worN=16000)
        # self.h *= signal.get_window('blackman', 16000)
        # plot_response(2, w, self.h, "Band-pass Filter")
        self.h = np.concatenate((self.h, self.h[::-1]), axis=None)
        self.fourier = np.real(self.fourier) * np.real(self.h) + np.imag(self.fourier) * np.imag(self.h)
        self.__fourier(False)
        self.reused = True

    # Функция, показывающая описание сигнала при команде print()
    def __str__(self):
        for i in range(len(self.head)):
            if self.head[i] == 'Mode':
                j = i + 2
                break
        else:
            return ''
        for i in range(len(self.head)):
            if self.head[i] == 'SWT':
                k = i + 3
                break
        else:
            return ''
        s = np.array(self.head[j:17] + self.head[19:34] + self.head[36:k], dtype='str').reshape(-1, 3)
        string = ''
        for j in s:
            string += j[0] + ': ' + j[1] + ' ' + j[2] + '\n'
        return string

    def __transform_adaptive(self, mode='from'):
        if mode == 'from':
            for i in range(self.max_points - self.size):
                self.adapted_signal[i:i + self.size] += self.filtered_signal[i]
        else:
            for i in range(self.max_points - self.size):
                self.prepared_signal[i] = self.signal[i:i + self.size]

    def adaptive_filter(self, window):
        self.filtered_signal = predict(window, self.prepared_signal)
        self.__transform_adaptive()
        self.__normalize()

    def __normalize(self):
        self.adapted_signal = (self.adapted_signal - np.min(self.adapted_signal)) / (
                    np.max(self.adapted_signal) - np.min(self.adapted_signal))
        self.normalized_signal = (self.normalized_signal - np.min(self.normalized_signal)) / (
                    np.max(self.normalized_signal) - np.min(self.normalized_signal))

def compare(N, *tm, plot=False):
    string = np.array(tm[0].head[8:17] + tm[0].head[19:34] + tm[0].head[36:51], dtype='str').reshape(1, 13, 3)
    s = '\t\t\t\t' + tm[0].name + '\t'
    for i in range(1, N):
        string = np.concatenate(
            (string, [np.array(tm[i].head[8:17] + tm[i].head[19:34] + tm[i].head[36:51], dtype='str').reshape(-1, 3)]),
            axis=0)
        s += tm[i].name + '\t'
    s += '\n'
    for i in range(len(string[0])):
        if len(np.unique(string[0:N, i, 1])) != 1:
            s += string[0][i][0] + ':\t' + str(string[0:N, i, 1]) + ' ' + string[0][i][2] + '\n'
    print(s)
    if plot:
        fig, ax = plt.subplots(1, 2)
        for i in range(N):
            ax[0].plot(tm[i].S)
            ax[1].plot(np.real(tm[i].f))
        fig.tight_layout()
        fig.show()


def plot_two(x, y):
    fig, ax_signal = plt.subplots(2, 2, sharex=True)
    ax_signal[0][0].plot(x.signal)
    ax_signal[1][0].plot(np.real(x.fourier))
    # ax_signal[2][0].plot(np.real(x.i))
    ax_signal[0][1].plot(y.signal)
    ax_signal[1][1].plot(np.real(y.fourier))
    # ax_signal[2][1].plot(np.real(y.i))
    fig.tight_layout()
    fig.show()


# Возвращает массив описывающий корреляцию между спектрами двум сигналов
# НЕОБХОДИМО! Автоматизировать метод получения наибольшего по дифференциалу экстремума.
def comp(x, y):
    f_x = np.abs(x.fourier[:16000]).reshape(100, -1)
    f_y = np.abs(y.fourier[:16000]).reshape(100, -1)
    res = np.zeros(100)
    for i in range(100):
        res[i] = signal.correlate(f_x[i], f_y[i], mode='valid')
    return res

    # Перевести в Picture


def iirdes(x, wp, ws, gpass, gstop, casc=False):
    x.iir(wp, ws, gpass, gstop)
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(15, 10)
    fig.suptitle(
        'Passband and stopband edge frequencies : {} - {}\n The maximum loss in the passband : {}\n The minimum attenuation in the stopband : {}'.format(
            wp, ws, gpass, gstop), fontsize=12)
    sg, fil, spec, res = ax
    sg.plot(np.real(x.signal))
    fil.plot(20 * np.log10(np.abs(x.h)))
    spec.plot(20 * np.log10(np.abs(x.fourier)))
    # spec.set_ylim(-25, 75)
    res.plot(np.abs(x.post_signal))
    p, _ = signal.find_peaks(x.post_signal, distance=1400, height=3.6)
    res.plot(p, np.abs(x.post_signal[p]), "x")
    sg.plot(p, x.signal[p], "x")
    plt.savefig(fname='Plots/Plot of {}, {}, {}, {} - ({}).jpg'.format(wp, ws, gpass, gstop, x.name))
    plt.show()
    # if input(print('Сохранить график отдельно?')) == 'y':
    #     plt.savefig(fname='Favorite/Plot of {}, {}, {}, {} - ({}).jpg'.format(wp, ws, gpass, gstop, x.name))