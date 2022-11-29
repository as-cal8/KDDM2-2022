import DataPreprocessing
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def rmTrend(df_train):
    # Differencing x_t - x_t-1
    # already done once with
    return df_train['diffMain'], df_train['diffOthers']

def rmSeasonality(df_train):
    # Seasonality
    # e.g. Lagged Differencing -> estimate s = 1/f
    fftt = fft((df_train['diffMain'] - np.mean(df_train['diffMain'])).values)
    plt.plot(np.abs(fftt))
    plt.title("FFT of main grid")
    plt.show()

def additiveModel(df_train):
    rmTrend(df_train)
    rmSeasonality(df_train)
    # focus on seasonality and trend!
    # for randomness use e.g. expected value ARIMA


if __name__ == "__main__":
    df_train, df_test = DataPreprocessing.dataPreprocesssing()
    additiveModel(df_train)
