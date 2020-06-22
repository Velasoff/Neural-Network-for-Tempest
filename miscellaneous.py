import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sklearn.metrics import mean_squared_error


def iir_design(wp, ws, gpass, gstop):
    b, a = signal.iirdesign(wp, ws, gpass, gstop)
    _, h = signal.freqs(b, a, worN=16000)
    h = np.concatenate((h, h[::-1]), axis=None)
    w = np.linspace(1, 10, 32000)
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(15, 10)
    fig.suptitle(
        'Passband and stopband edge frequencies : {} - {}\n The maximum loss in the passband : {}\n The minimum attenuation in the stopband : {}'.format(
            wp, ws, gpass, gstop), fontsize=12)
    r, im, abs, ang = ax
    r.plot(w, 20 * np.log10(np.real(h)))
    im.plot(w, 20 * np.log10(np.imag(h)))
    abs.plot(w, 20 * np.log10(np.abs(h)))
    ang.plot(w, 20 * np.log10(np.angle(h)))
    plt.savefig(fname='Plots_of_design/Plot of {}, {}, {}, {}.jpg'.format(wp, ws, gpass, gstop))


def plot_adapted(data, MLP):
    prepared = np.empty([31900, 100])
    for i in range(31900):
        prepared[i] = data[i:i + 100]
    adapt = MLP.predict(prepared)
    adapted = np.zeros(data.shape)
    for i in range(31900):
        adapted[i:i + 100] += adapt[i]
    plt.figure(figsize=(8, 8), dpi=80)
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.subplot(3, 1, 2)
    plt.plot(adapted)
    f1 = [mean_squared_error(prepared[i], adapt[i]) for i in range(31900)]
    plt.subplot(3, 1, 3)
    plt.plot(f1)
    print(mean_squared_error(data, adapted))


def make_some_noise(gaus=1.,
                    impuls=1., num_impuls=20,
                    radio_impuls=1., num_radio=10,
                    amp_freq=[(1., 10e-10)],
                    amp=1):
    noize = np.zeros(32000)
    if amp == 0:
        return noize
    if gaus > 0:
        noize += np.random.normal(scale=gaus / 3, size=noize.shape)
    for amp_sinus, freq_sinus in amp_freq:
        if amp_sinus > 0:
            noize += np.fromfunction(lambda i: amp_sinus * np.sin((2 / 32000) * np.pi * freq_sinus * i), (32000,),
                                     dtype='float')
    if impuls > 0:
        freq = 32000 // num_impuls
        for i in range(num_impuls):
            j = random.randrange(i * freq, (i + 1) * freq)
            width = int(np.random.normal(loc=25, scale=10) % 20)
            try:
                noize[j:j + width] += np.random.normal(loc=impuls, scale=0.2 * impuls, size=width)
            except ValueError:
                pass
    if radio_impuls > 0 and num_radio > 0:
        freq = 32000 // num_radio
        b, a = signal.butter(4, 0.2)
        imp = signal.unit_impulse(25)
        for i in range(num_radio):
            j = random.randrange(i * freq, (i + 1) * freq)
            try:
                noize[j:j + 25] += 5 * radio_impuls * signal.lfilter(b, a, imp)
            except ValueError:
                pass
    noize *= amp
    return noize


def get_text_desc_mass(min_SN=0, max_SN=0, iterations=0, params={}):
    harmonic_txt = ""
    for i, j in params["amp_freq"]:
        harmonic_txt += "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; " \
                        "-qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">	 " \
                        f"</span><span style=\" font-size:8pt; font-style:italic;\">{100 * i}	{j}</span></p> "
    txt = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\"> <html><head>" \
          "<meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">p, li { white-space: pre-wrap; }</style></head>" \
          "<body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">" \
          "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; " \
          "-qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">" \
          "Свойства выбранного массива:</span></p><p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; " \
          "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt; " \
          "font-weight:600;\"><br /></p><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; " \
          "margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; " \
          "font-weight:600;\">Количество выборок: " + str(iterations) + "</span></p><p style=\"-qt-paragraph-type" \
                                                                        ":empty; margin-top:0px; " \
                                                                        "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt; " \
                                                                        "font-weight:600;\"><br /></p><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; " \
                                                                        "margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; " \
                                                                        "font-weight:600;\">Диапазон: " + str(
        min_SN) + " - " + str(max_SN) + "</span><span style=\" " \
                                        "font-size:8pt; font-style:italic;\">Дб</span></p><p " \
                                        "style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; " \
                                        "-qt-block-indent:0; text-indent:0px; font-size:8pt; font-style:italic;\"><br /></p><p align=\"center\" style=\" " \
                                        "margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; " \
                                        "text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Свойства компонентов:</span></p><p " \
                                        "align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; " \
                                        "margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p><p " \
                                        "style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; " \
                                        "text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Гауссовый шум: 	</span><span style=\" " \
                                        "font-size:8pt; font-weight:600;\">Мощность: " + str(
        params["gaus"] * 100) + "</span><span style=\" font-size:8pt; " \
                                "font-style:italic;\">%</span></p><p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; " \
                                "margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt; " \
                                "font-weight:600;\"><br /></p><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; " \
                                "margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; " \
                                "font-weight:600;\">Радиоимпульсы:	</span><span style=\" font-size:8pt; font-weight:600;\">Мощность: " \
          + str(params["radio_impuls"] * 100) + \
          "</span><span style=\" font-size:8pt; font-style:italic;\">%</span></p><p style=\" margin-top:0px; " \
          "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" " \
          "font-size:8pt; font-weight:600;\">		Количество импульсов: " + str(
        params["num_radio"]) + "</span></p><p " \
                               "style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; " \
                               "-qt-block-indent:0; text-indent:0px; font-size:8pt; font-weight:600;\"><br /></p><p style=\" margin-top:0px; " \
                               "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" " \
                               "font-size:9pt; font-weight:600;\">Импульсы:		</span><span style=\" font-size:8pt; " \
                               "font-weight:600;\">Мощность: " + str(
        params["impuls"] * 100) + "</span><span style=\" font-size:8pt; " \
                                  "font-style:italic;\">%</span></p><p style=\" " \
                                  "margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; " \
                                  "text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">		Количество импульсов: " \
          + str(params["num_impuls"]) + "</span></p><p " \
                                        "style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; " \
                                        "-qt-block-indent:0; text-indent:0px; font-size:8pt; font-weight:600;\"><br /></p><p style=\" margin-top:0px; " \
                                        "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" " \
                                        "font-size:8pt; font-weight:600;\">Гармонические компоненты:</span></p><p style=\" margin-top:0px; " \
                                        "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" " \
                                        "font-size:8pt; font-weight:600;\">	Мощность: 	Частота:</span></p><p style=\" margin-top:0px; " \
                                        "margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" " \
                                        "font-size:8pt; font-weight:600;\">	 </span><span style=\" font-size:8pt; font-style:italic;\">%	" \
                                        "Гц</span></p>" + harmonic_txt + "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; " \
                                                                         "margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt; font-weight:600;\"><br " \
                                                                         "/></p></body></html> "
    return txt


def interpolate_core(mass: np.ndarray, core_range: int):
    n = len(mass)
    mass_new = np.empty(n)
    for i in range(n):
        if i < core_range:
            mass_new[i] = mass[: i + core_range].mean()
        elif n - i < core_range:
            mass_new[i] = mass[i - core_range :].mean()
        else:
            mass_new[i] = mass[i - core_range : i + core_range].mean()
    return mass_new

