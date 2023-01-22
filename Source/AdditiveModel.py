# Sources used: https://towardsai.net/p/machine-learning/seasonality-detection-with-fast-fourier-transform-fft-and-python
#               https://github.com/netsatsawat/tutorial_fft_seasonality_detection/blob/master/FFT%20Tutorial.ipynb
#               https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
import numpy as np

import DataPreprocessing
from scipy.fftpack import rfft, rfftfreq
import scipy.signal.windows as window
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose
from Evaluation import *
import os
from scipy.optimize import leastsq
from statsmodels.tsa.arima.model import ARIMA

def plotDataAndWindow(time_: np.ndarray,
                      val_orig: pd.core.series.Series,
                      val_window: pd.core.series.Series):
    plt.figure(figsize=(14, 10))
    plt.plot(time_, val_orig, label='raw')
    plt.plot(time_, val_window, label='windowed time')
    plt.legend()
    plt.show()

def plotFTResults(val_orig_psd: np.ndarray,
                  val_widw_psd: np.ndarray,
                  ft_smpl_freq: np.ndarray,
                  pos: int = 2, annot_mode: bool = True
                  ):
    """
    For PSD graph, the first few points are removed because it represents the baseline (or mean)
    """
    plt.figure(figsize=(14, 10))
    plt.plot(ft_smpl_freq[pos:], val_orig_psd[pos:], label='psd original value')
    plt.plot(ft_smpl_freq[pos:], val_widw_psd[pos:], label='psd windowed value')
    plt.xlabel('frequency (1/h)')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return

def FFTAnalysis(data_mainGrid, ignore_plots):
    # window function
    if type(data_mainGrid) == pd.Series:
        data_mainGrid = data_mainGrid[5408:23031]
        data_window = ((data_mainGrid - np.median(data_mainGrid)) * window.blackman(len(data_mainGrid))).values
    else:
        data_window = (data_mainGrid - np.median(data_mainGrid)) * window.hamming(len(data_mainGrid), sym=False)

    #data_window = data_mainGrid
    Fs = 1  # sampling rate
    fftt = np.abs(rfft(data_window)) / len(data_window)
    amplitudes = 2 / len(data_window) * np.abs(fftt)
    ind = np.argpartition(fftt, -10)[-10:]
    ind = ind[-5:]
    topFFTT = np.flip(np.sort(fftt[ind]))
    ind = []
    for val in topFFTT:
        ind.append(int(np.where(fftt == val)[0]))
    freq = rfftfreq(int(len(data_window)), d=1 / Fs)
    topFreq = np.unique(freq[ind])
    topFreq = topFreq[topFreq != 0.]  # 0 is baseline, useless
    topHourSeason = np.sort(1 / topFreq)
    topDaySeason = topHourSeason / 24
    if not ignore_plots:
        print("Top 4 found frequencies [1/h]: ")
        print(topFreq)
        print(" \t\t [hours]: ")
        print(topHourSeason)
        print(" \t\t [days]: ")
        print(topDaySeason)

        plotDataAndWindow(df_train['timestamps'][5408:23031], data_mainGrid, data_window)
        plotFTResults(fftt, abs(rfft(data_window) / len(data_window)), freq)
    plt.plot(df_train['timestamps'][5408:23031], df_train.mainGrid[5408:23031])
    return topHourSeason, topDaySeason

def FFTAnalysisAny(data, ignore_plots=False):
    data_window = data
    Fs = 1  # sampling rate
    fftt = np.abs(rfft(data_window)) / len(data_window)
    ind = np.argpartition(fftt, -10)[-10:]
    ind = ind[-5:]
    freq = rfftfreq(int(len(data_window)), d=1 / Fs)
    topFreq = np.unique(freq[ind])
    topFreq = topFreq[topFreq != 0.]  # 0 is baseline, useless
    topHourSeason = np.sort(1 / topFreq)
    topDaySeason = topHourSeason / 24
    if not ignore_plots:
        print("Top 4 found frequencies [1/h]: ")
        print(topFreq)
        print(" \t\t [hours]: ")
        print(topHourSeason)
        print(" \t\t [days]: ")
        print(topDaySeason)

        plotFTResults(fftt, abs(rfft(data_window) / len(data_window)), freq)
    return topHourSeason, topDaySeason


def additiveModel(df_train, t=None, ignore_plots=False):
    data_mainGrid = df_train['mainGrid']

    topHourSeason, topDaySeason = FFTAnalysis(data_mainGrid, ignore_plots)

    sd_0 = seasonal_decompose(df_train['mainGrid'].values, period=int(np.round(topHourSeason[0])), model="additive")

    sd_1 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_0.seasonal).reshape(-1, 1),
                              period=int(np.round(topHourSeason[1])), model="additive")
    sd_2 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_1.seasonal).reshape(-1, 1),
                              period=int(np.round(topHourSeason[2])), model="additive")
    sd_3 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_2.seasonal).reshape(-1, 1),
                              period=int(np.round(topHourSeason[3])), model="additive", extrapolate_trend='freq')


    if not ignore_plots:
        # drawing figure with subplots, predefined size and resolution
        f, axes = plt.subplots(8, 1, figsize=(130, 8))
        # setting figure title and adjusting title position and size
        range_start = 5000
        range_end = 12500
        plt.suptitle('Summary of seasonal decomposition', y=0.92)
        axes[0].plot(sd_3.trend)
        axes[0].set_title('Trend component')
        #axes[0].set_ylim([85, 280])
        axes[1].plot(sd_0.seasonal[range_start:range_end])
        axes[1].set_title(str(np.round(topDaySeason[0], 2)) + ' day season')
        axes[1].set_ylim([-50, 50])
        axes[2].plot(sd_1.seasonal[range_start:range_end])
        axes[2].set_title(str(np.round(topDaySeason[1], 2)) + ' day season')
        axes[2].set_ylim([-50, 50])
        axes[3].plot(sd_2.seasonal[range_start:range_end])
        axes[3].set_title(str(np.round(topDaySeason[2], 2)) + ' day  season')
        axes[3].set_ylim([-50, 50])
        axes[4].plot(sd_3.seasonal)
        axes[4].set_title(str(np.round(topDaySeason[3], 2)) + ' day  season')
        axes[4].set_ylim([-50, 50])
        axes[5].plot(sd_3.resid[range_start:range_end])
        axes[5].set_title('resid')
        axes[5].set_ylim([-50, 50])
        # plot sum of seasonal decomposition + offset (=mean of trend)
        axes[6].plot(sd_0.seasonal[range_start:range_end] + sd_1.seasonal[range_start:range_end] + sd_2.seasonal[
                                                                                                   range_start:range_end] + sd_3.seasonal[
                                                                                                                            range_start:range_end] + sd_3.trend[
                                                                                                   range_start:range_end])
        axes[6].set_title('sum of seasons')
        axes[6].plot(pd.Series.to_numpy(df_train["mainGrid"][range_start:range_end]), "r")
        plt.show()

    # save residuals/ noise to csv for further use
    path = os.getcwd() + str("\\Data\\noise.csv")
    np.savetxt(path, sd_3.resid, delimiter=",")

    '''
    Wave functions to model found seasons with
    '''
    def f0(t, A, b, C):
        return A * np.sin(C *2 * np.pi / topHourSeason[0] * t + b)

    def f0x(t, params):
        return f0(t, params[0], params[1], params[2])

    def f1(t, A, b, C):
        return A * np.sin(C *2 * np.pi / topHourSeason[1] * t + b)

    def f1x(t, params):
        return f1(t, params[0], params[1], params[2])

    '''
    For predicting more complicated season, repeating pattern was used.
    '''
    def f2x(t):
        prediction = []
        for ti in t:
            pred_i = ti-int(np.round(topHourSeason[2]))
            i = 2
            while pred_i >= len(sd_2.seasonal):
                pred_i = ti - (i * int(np.round(topHourSeason[2])))
                i += 1
            prediction.append(sd_2.seasonal[pred_i])
        return prediction

    def f3x(t):
        '''
        moving average of yearly season
        :param t:
        :param params:
        :return:
        '''
        prediction = []
        for ti in t:
            pred_i = ti - int(np.round(topHourSeason[3]))
            i = 2
            while pred_i >= len(sd_3.seasonal):
                pred_i = ti - (i * int(np.round(topHourSeason[3])))
                i += 1
            prediction.append(np.mean(sd_3.seasonal[pred_i-100:pred_i]))
        return prediction

    # Functions to model trend component
    def f4(t, A, d):
        return A * t + d

    def f4x(t, params):
        return f4(t, params[0], params[1])

    x = df_train.iloc[:, 0].values
    para0, para0_cov = curve_fit(f0, x, sd_0.seasonal)
    para1, para1_cov = curve_fit(f1, x, sd_1.seasonal)
    para4, para4_cov = curve_fit(f4, x, sd_3.trend) # trend component

    x_plt = x[10000:20000]

    if not ignore_plots:
        plt.figure()
        plt.title("Fitted curve for season 3")
        plt.plot(x_plt, f4x(x_plt, para4), label="fit")
        plt.plot(x_plt, sd_3.trend[10000:20000], label="sd_3")
        plt.legend()
        plt.show()

    a = 1

    def modelAdditive(t):
        """
        Additive model
        :param t: time stamps [hour indexes] to predict
        :return: prediction
        """
        return f0x(t, para0) + f1x(t, para1) + f2x(t) + f3x(t) + f4x(t, para4)

    if not ignore_plots:
        t_test = df_test.iloc[:, 0].values
        plotActualVsPred(df_test['mainGrid'], modelAdditive(t_test), 'mainGrid')
        evaluationMetrics(modelAdditive(t_test), df_test["mainGrid"])

    if t is not None:
        return modelAdditive(t)

def getAdditiveModel(data_train, t_pred):
    """
    Returns the prediction values of the additive model for each time in t [index of hour]
    Model is created with data_train.

    :param data_train: training data to create additive model from
    :param t_pred: timestamps on which to make predictions
    :return: list of predictions, same length as t_pred
    """
    return additiveModel(data_train, t_pred, ignore_plots=True)


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    t = df_test.iloc[:, 0].values
    additiveModel(df_train, t, ignore_plots=False)