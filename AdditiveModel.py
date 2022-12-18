from scipy.signal.windows import blackman

import DataPreprocessing
import numpy as np
from scipy.fftpack import rfft, rfftfreq
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from statsmodels.tsa.seasonal import STL, seasonal_decompose

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
    # Differencing x_t - x_t-1 ???
    return df_train

def rmSeasonality(df_train):
    # Seasonality
    detre = (df_train['detrendMain'] - np.median(df_train['detrendMain'])).values

    val_widw = (detre - np.median(detre)) * blackman(len(detre))
    Fs = 1 # sampling rate
    fftt = np.abs(rfft(detre)) / len(detre)
    ind = np.argpartition(fftt, -6)[-6:]
    print("Top 4 found frequencies [1/h]: ")
    freq = rfftfreq(int(len(detre)), d=1/Fs)
    top5freq = np.unique(freq[ind])
    print(top5freq)
    print(" \t\t [hours]: ")
    top5hourSeason = np.sort(1 / top5freq)
    print(top5hourSeason)
    print(" \t\t [days]: ")
    top5daySeason = top5hourSeason / 24
    print(top5daySeason)

    plt.stem(freq, fftt)
    plt.show()
    sd_0 = seasonal_decompose(df_train['detrendMain'].values, period=int(top5hourSeason[0]))
    sd_1 = seasonal_decompose(df_train['detrendMain'].values.reshape(-1, 1) - np.array(sd_0.seasonal).reshape(-1, 1), period=int(top5hourSeason[1]))
    sd_2 = seasonal_decompose(df_train['detrendMain'].values.reshape(-1, 1) - np.array(sd_1.seasonal).reshape(-1, 1), period=int(top5hourSeason[2]))
    sd_3 = seasonal_decompose(df_train['detrendMain'].values.reshape(-1, 1) - np.array(sd_2.seasonal).reshape(-1, 1), period=int(top5hourSeason[3]))
    sd_4 = seasonal_decompose(df_train['detrendMain'].values.reshape(-1, 1) - np.array(sd_3.seasonal).reshape(-1, 1), period=int(top5hourSeason[4]))
    # drawing figure with subplots, predefined size and resolution
    f, axes = plt.subplots(7, 1, figsize=(80, 8))
    # setting figure title and adjusting title position and size
    range_start = 6000
    range_end = 10000
    plt.suptitle('Summary of seasonal decomposition', y=0.92)
    axes[0].plot(sd_4.trend[range_start:range_end])
    axes[0].set_title('Trend component')
    axes[0].set_ylim([-50,50])
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
    axes[5].plot(sd_4.seasonal[range_start:range_end])
    axes[5].set_title(str(np.round(top5daySeason[4], 2)) + ' day  season')
    axes[4].set_ylim([-50,50])
    axes[6].plot(sd_4.resid[range_start:range_end])
    axes[6].set_title('resid')
    axes[6].set_ylim([-50,50])

    plt.show()

def additiveModel(df_train):
    df = rmTrend(df_train)
    rmSeasonality(df)
    # focus on seasonality and trend!
    # for randomness use e.g. expected value ARIMA


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    additiveModel(df_train)
