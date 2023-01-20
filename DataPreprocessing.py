import pandas as pd
import numpy as np
import DataLoading
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split

# header = {"mainGrid", "otherGrids", "hourOfDay"}
# three year power supply records from 1995 to 1998
def dataPreprocesssing():
    df = pd.read_csv(DataLoading.path)

    # create timestamps
    start = datetime.datetime(1995, 1, 1, 0)
    timestamp_list = [start + datetime.timedelta(hours=x) for x in range(len(df))]
    df['timestamps'] = timestamp_list

    # mistake in hours numbering hour 6 = 1248 instances
    #                                 9 = 1246 instances
    wrongHoursIndex = 24969
    #for i in np.arange(len(df)-1):
    #    if df['hourOfDay'][i]+1 != df['hourOfDay'][i + 1] and df['hourOfDay'][i] != 23:
    #        print("wtf")
    df.at[wrongHoursIndex, "hourOfDay"] = 9

    df_train, df_test = train_test_split(df, test_size=6000, shuffle=False) # ~20% test split
    # hourly change in consumption
    diffMain = np.append(np.array(df_train['mainGrid']), 0) - np.append(0,np.array(df_train['mainGrid']))
    diffMain = diffMain[0:-1]
    diffMain[0] = 0
    diffOthers = np.append(np.array(df_train['otherGrids']), 0) - np.append(0,np.array(df_train['otherGrids']))
    diffOthers = diffOthers[0:-1]
    diffOthers[0] = 0

    df_train['diffMain'] = diffMain
    df_train['diffOthers'] = diffOthers

    df_hourDiffMain = pd.DataFrame()
    df_hourDiffOther = pd.DataFrame()
    for hour in np.arange(24):
        hourDiffList = df_train.loc[df_train['hourOfDay'] == hour]['diffMain'].values
        df_hourDiffMain[str(hour)] = hourDiffList
        hourDiffList = df_train.loc[df_train['hourOfDay'] == hour]['diffOthers'].values
        df_hourDiffOther[str(hour)] = hourDiffList

    # figure(figsize=(22, 5), dpi=80, linewidth=5)
    # plt.title("Main Grid boxplt change of every hour through time")
    # df_hourDiffMain.boxplot()
    # figure(figsize=(22, 5), dpi=80, linewidth=5)
    # plt.title("Other Grids boxplt change of every hour through time")
    # df_hourDiffOther.boxplot()
    # plt.show()
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.boxplot(diffMain)
    # ax1.set_title("main Grid")
    # ax2.boxplot(diffOthers)
    # ax2.set_title("other Grids")
    # plt.show()
    #
    # figure(figsize=(90, 16), dpi=80)
    # plt.plot(df_train['timestamps'], df_train['mainGrid'], 'r-')
    # #plt.plot(df_train['timestamps'], df_train['otherGrids'], 'b-')
    # plt.title('Power constumption')
    # plt.xlabel('time', fontsize=14)
    # plt.ylabel('consumption', fontsize=14)
    # plt.show()
    #
    # figure(figsize=(80, 5), dpi=80, linewidth=1)
    # plt.plot(df_train['timestamps'], df_train['diffMain'], 'r-')
    # #plt.plot(df_train['timestamps'], df_train['diffOthers'], 'b-')
    # plt.title('Power constumption diff')
    # plt.xlabel('time', fontsize=14)
    # plt.ylabel('consumption', fontsize=14)
    # plt.show()

    return df_train, df_test


if __name__ == "__main__":
    dataPreprocesssing()


'''
Check weak stationary:
    mean constant
    variance finite, few outliers
    autocovariance only changes with relative lag

Additive model: !!!! FOCUS ON TREND AND SEASONALITY -> INVARIANT !!!!
    data = trend + seasonality + randomness
    -> Detrend (Linear Filter, Regression, Differencing (preferred))
    -> remove Seasonality (Low pass filter, Lagged differencing)
        -> also try Season length estimator -> Autocorrelation analysis (or Spectral analysis)

Make Data Locally stationary -> Window Functions
    -> to then apply univariate Models
        -> exp smoothing, ARMA, ARIMA, GARCH
    -> or ML models (RNN, LSTM)
'''
