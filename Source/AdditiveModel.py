# Sources used: https://towardsai.net/p/machine-learning/seasonality-detection-with-fast-fourier-transform-fft-and-python
#               https://github.com/netsatsawat/tutorial_fft_seasonality_detection/blob/master/FFT%20Tutorial.ipynb

import DataPreprocessing
from scipy.fftpack import rfft, rfftfreq
import scipy.signal.windows as window
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose
from Evaluation import *
import os

def plotDataAndWindow(time_: np.ndarray,
                      val_orig: pd.core.series.Series,
                      val_window: pd.core.series.Series):
    plt.figure(figsize=(14, 10))
    plt.plot(time_, val_orig, label='raw')
    plt.plot(time_, val_window, label='windowed time')
    plt.legend()
    plt.show()
    return

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

def additiveModel(df_train, t=None, ignore_plots=False):
    data_mainGrid = df_train['mainGrid']
    # window function
    data_window = ((data_mainGrid - np.median(data_mainGrid)) * window.hamming(len(data_mainGrid), sym=False)).values

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

        plotDataAndWindow(df_train['timestamps'], df_train['mainGrid'], data_window)
        plotFTResults(fftt, abs(rfft(data_window) / len(data_window)), freq)

    sd_0 = seasonal_decompose(df_train['mainGrid'].values, period=int(topHourSeason[0]))
    sd_1 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_0.seasonal).reshape(-1, 1),
                              period=int(topHourSeason[1]))
    sd_2 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_1.seasonal).reshape(-1, 1),
                              period=int(topHourSeason[2]))
    sd_3 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_2.seasonal).reshape(-1, 1),
                              period=int(topHourSeason[3]), extrapolate_trend='freq')
    if not ignore_plots:
        # drawing figure with subplots, predefined size and resolution
        f, axes = plt.subplots(8, 1, figsize=(130, 8))
        # setting figure title and adjusting title position and size
        range_start = 10000
        range_end = 12500
        plt.suptitle('Summary of seasonal decomposition', y=0.92)
        axes[0].plot(sd_3.trend[range_start:range_end])
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
        axes[4].plot(sd_3.seasonal[range_start:range_end])
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
        axes[7].set_title('main grid data')
        plt.show()

    # save residuals/ noise to csv for further use
    path = os.getcwd() + str("\\Data\\noise.csv")
    np.savetxt(path, sd_3.resid, delimiter=",")

    # wave functions to model found seasons
    def f0(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / topHourSeason[0] * t + b) + C * np.cos(
            2 * np.pi / topHourSeason[0] * t + d) + E

    def f0x(t, params):
        return f0(t, params[0], params[1], params[2], params[3], params[4])

    def f1(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / topHourSeason[1] * t + b) + C * np.cos(
            2 * np.pi / topHourSeason[1] * t + d) + E

    def f1x(t, params):
        return f1(t, params[0], params[1], params[2], params[3], params[4])

    def f2(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / topHourSeason[2] * t + b) + C * np.cos(
            2 * np.pi / topHourSeason[2] * t + d) + E

    def f2x(t, params):
        return f2(t, params[0], params[1], params[2], params[3], params[4])

    def f3(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / topHourSeason[3] * t + b) + C * np.cos(
            2 * np.pi / topHourSeason[3] * t + d) + E

    def f3x(t, params):
        return f3(t, params[0], params[1], params[2], params[3], params[4])

    # Functions to model trend component
    def f4(t, A, d):
        return A * t + d

    def f4x(t, params):
        return f4(t, params[0], params[1])

    x = df_train.iloc[:, 0].values
    para0, para0_cov = curve_fit(f0, x, sd_0.seasonal)
    para1, para1_cov = curve_fit(f1, x, sd_1.seasonal)
    para2, para2_cov = curve_fit(f2, x, sd_2.seasonal)
    para3, para3_cov = curve_fit(f3, x, sd_3.seasonal)
    para4, para4_cov = curve_fit(f4, x, sd_3.trend) # trend component

    if not ignore_plots:
        x_plt = x[1000:1500]
        plt.plot(x_plt, f0(x_plt, para0[0], para0[1], para0[2], para0[3], para0[4]))
        plt.plot(x_plt, sd_0.seasonal[1000:1500])
        plt.show()

    # FINAL ADDITIVE MODEL:
    def model_additive(t):
        return f0x(t, para0) + f1x(t, para1) + f2x(t, para2) + f3x(t, para3) + f4x(t, para4)

    if not ignore_plots:
        t_test = df_test.iloc[:, 0].values
        plt.figure()
        plt.plot(t_test, model_additive(t_test), )
        plt.plot(t_test, df_test['mainGrid'])
        plt.show()

        evaluationMetrics(model_additive(t_test), df_test["mainGrid"])

    if t is not None:
        return model_additive(t)

'''
- data_train, training dataset, where seasons are extracted. (from DataPreprocessing)
- t, array of timestamps for which to calculate estimates

returns array of estimates
'''
def getAdditiveModel(data_train, t):
    return additiveModel(data_train, t, ignore_plots=True)


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    t = df_test.iloc[:, 0].values
    additiveModel(df_train, t, ignore_plots=False)
