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
    Fs = 24 # sampling rate
    fftt = np.abs(rfft(detre)) / len(detre)
    freq = rfftfreq(int(len(df_train.index)), d=1/Fs)
    #plt.stem(freq, fftt)
    #plt.show()
    sd_24 = seasonal_decompose(detre, period=24)
    sd_48 = seasonal_decompose(detre.reshape(-1,1) - np.array(sd_24.seasonal).reshape(-1, 1), period=48)
    # drawing figure with subplots, predefined size and resolution
    f, axes = plt.subplots(4, 1, figsize=(80, 8))
    # setting figure title and adjusting title position and size
    plt.suptitle('Summary of seasonal decomposition', y=0.92)
    axes[0].plot(sd_48.trend[:1000])
    axes[0].set_title('Trend component')
    axes[1].plot(sd_24.seasonal[:1000])
    axes[1].set_title('Daily component')
    axes[2].plot(sd_48.seasonal[:1000])
    axes[2].set_title('2 Days component')
    axes[3].plot(sd_48.resid[:1000])
    axes[3].plot('2 Days resid')
    plt.show()

def additiveModel(df_train):
    df = rmTrend(df_train)
    rmSeasonality(df)
    # focus on seasonality and trend!
    # for randomness use e.g. expected value ARIMA


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    additiveModel(df_train)
