
import DataPreprocessing
import numpy as np
from scipy.fftpack import rfft, rfftfreq
import scipy.signal.windows as window
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import pandas as pd

def plot_ori_window(time_: np.ndarray,
                    val_orig: pd.core.series.Series,
                    val_window: pd.core.series.Series):
    plt.figure(figsize=(14, 10))
    plt.plot(time_, val_orig, label='raw')
    plt.plot(time_, val_window, label='windowed time')
    plt.legend()
    plt.show()
    return


def plot_ft_result(val_orig_psd: np.ndarray,
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

def rmTrend(df_train):
    regr = linear_model.LinearRegression()
    x = df_train.iloc[:, 0].values
    y = df_train['mainGrid'].values
    x = x.reshape(np.size(x), 1)
    y = y.reshape(np.size(y), 1)
    regr.fit(x,y)
    mainPred = regr.predict(df_train['mainGrid'].values.reshape(np.size(x), 1))
    mainDetrend = df_train['mainGrid'].values.reshape(np.size(x), 1)-mainPred
    df_train['detrendMain'] = mainDetrend
    return df_train

def rmSeasonality(df_train):
    # Seasonality
    detre = (df_train['mainGrid'] - np.median(df_train['mainGrid'])).values

    # window function
    _val_orig = df_train['mainGrid']
    _val_widw = ((_val_orig - np.median(_val_orig)) * window.hamming(len(_val_orig), sym=False)).values
    detre = _val_widw

    Fs = 1 # sampling rate
    fftt = np.abs(rfft(detre)) / len(detre)
    ind = np.argpartition(fftt, -10)[-10:]
    ind = ind[-5:]
    print("Top 4 found frequencies [1/h]: ")
    freq = rfftfreq(int(len(detre)), d=1/Fs)
    top5freq = np.unique(freq[ind])
    top5freq = top5freq[top5freq != 0.] # 0 is baseline, useless
    print(top5freq)
    print(" \t\t [hours]: ")
    top5hourSeason = np.sort(1 / top5freq)
    print(top5hourSeason)
    print(" \t\t [days]: ")
    top5daySeason = top5hourSeason / 24
    print(top5daySeason)
    #top5hourSeason = np.array([12, 24, 84.0763312, 7.006369, 275., 366.6666666])
    #top5daySeason = top5hourSeason / 24

    #plot_ori_window(df_train['timestamps'], df_train['mainGrid'], _val_widw)
    #plot_ft_result(fftt, abs(rfft(_val_widw)/len(_val_widw)), freq)
    #plt.stem(freq, fftt)
    #plt.stem(freq, rfft(_val_widw.values))
    #plt.show()
    sd_0 = seasonal_decompose(df_train['mainGrid'].values, period=int(top5hourSeason[0]))
    sd_1 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_0.seasonal).reshape(-1, 1), period=int(top5hourSeason[1]))
    sd_2 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_1.seasonal).reshape(-1, 1), period=int(top5hourSeason[2]))
    sd_3 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_2.seasonal).reshape(-1, 1), period=int(top5hourSeason[3]))
    #sd_4 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_3.seasonal).reshape(-1, 1), period=int(top5hourSeason[4]))
    #sd_5 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_4.seasonal).reshape(-1, 1), period=int(top5hourSeason[5]))
    #sd_6 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_5.seasonal).reshape(-1, 1), period=int(top5hourSeason[6]))
    #sd_7 = seasonal_decompose(df_train['mainGrid'].values.reshape(-1, 1) - np.array(sd_6.seasonal).reshape(-1, 1), period=int(top5hourSeason[7]))

    offset = np.nanmean(sd_3.trend)

    # drawing figure with subplots, predefined size and resolution
    f, axes = plt.subplots(8, 1, figsize=(130, 8))
    # setting figure title and adjusting title position and size
    range_start = 10000
    range_end = 12500
    plt.suptitle('Summary of seasonal decomposition', y=0.92)
    axes[0].plot(sd_3.trend[range_start:range_end])
    axes[0].set_title('Trend component')
    axes[0].set_ylim([-25,25])
    axes[1].plot(sd_0.seasonal[range_start:range_end])
    axes[1].set_title(str(np.round(top5daySeason[0], 2)) + ' day season')
    axes[1].set_ylim([-50,50])
    axes[2].plot(sd_1.seasonal[range_start:range_end])
    axes[2].set_title(str(np.round(top5daySeason[1], 2)) + ' day season')
    axes[2].set_ylim([-50,50])
    axes[3].plot(sd_2.seasonal[range_start:range_end])
    axes[3].set_title(str(np.round(top5daySeason[2], 2)) + ' day  season')
    axes[3].set_ylim([-50,50])
    axes[4].plot(sd_3.seasonal[range_start:range_end])
    axes[4].set_title(str(np.round(top5daySeason[3], 2)) + ' day  season')
    axes[4].set_ylim([-50,50])
    axes[5].plot(sd_3.resid[range_start:range_end])
    axes[5].set_title('resid')
    axes[5].set_ylim([-50, 50])
    # plot sum of seasonal decomposition + offset (=mean of trend)
    axes[6].plot(sd_0.seasonal[range_start:range_end]+sd_1.seasonal[range_start:range_end]+sd_2.seasonal[range_start:range_end]+sd_3.seasonal[range_start:range_end] + offset)
    axes[6].set_title('sum of seasons')
    axes[7].plot(df_train["mainGrid"][range_start:range_end])
    axes[7].set_title('main grid data')
    plt.show()

    # wave functions to model found seasons
    def f0(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / top5hourSeason[0] * t + b) + C * np.cos(2 * np.pi / top5hourSeason[0] * t + d) + E
    def f0x(t, params):
        return f0(t, params[0], params[1], params[2], params[3], params[4])

    def f1(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / top5hourSeason[1] * t + b) + C * np.cos(2 * np.pi / top5hourSeason[1] * t + d) + E
    def f1x(t, params):
        return f1(t, params[0], params[1], params[2], params[3], params[4])

    def f2(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / top5hourSeason[2] * t + b) + C * np.cos(2 * np.pi / top5hourSeason[2] * t + d) + E
    def f2x(t, params):
        return f2(t, params[0], params[1], params[2], params[3], params[4])

    def f3(t, A, b, C, d, E):
        return A * np.sin(2 * np.pi / top5hourSeason[3] * t + b) + C * np.cos(2 * np.pi / top5hourSeason[3] * t + d) + E
    def f3x(t, params):
        return f3(t, params[0], params[1], params[2], params[3], params[4])

    x = df_train.iloc[:, 0].values
    para0, para0_cov = curve_fit(f0, x, sd_0.seasonal)
    para1, para1_cov = curve_fit(f1, x, sd_1.seasonal)
    para2, para2_cov = curve_fit(f2, x, sd_2.seasonal)
    para3, para3_cov = curve_fit(f3, x, sd_3.seasonal)
    x_plt = x[1000:1500]
    # plt.plot(x_plt, f0(x_plt, para0[0], para0[1], para0[2], para0[3], para0[4]))
    # plt.plot(x_plt, sd_0.seasonal[1000:1500])
    # plt.show()

    # FINAL ADDITIVE MODEL:
    def model_additive(t):
        return f0x(t, para0) + f1x(t, para1) + f2x(t, para2) + f3x(t, para3) + offset

    t = df_test.iloc[:, 0].values
    plt.figure()
    plt.plot(t, model_additive(t), )
    plt.plot(t, df_test['mainGrid'])
    plt.show()

    mse = metrics.mean_squared_error(df_test["mainGrid"], model_additive(t))
    print("============================")
    print("Additive model MSE:")
    print(" \t\t\t " + str(np.round(mse,4)))
    print("============================")



def additiveModel(df_train):
    #df = rmTrend(df_train)
    rmSeasonality(df_train)
    # focus on seasonality and trend!
    # for randomness use e.g. expected value ARIMA


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    additiveModel(df_train)
