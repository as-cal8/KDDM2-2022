import pandas as pd
import numpy as np
import DataLoading
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import datetime

# header = {"mainGrid", "otherGrids", "hourOfDay"}
# three year power supply records from 1995 to 1998
def main():
    df = pd.read_csv(DataLoading.path)

    # create timestamps
    start = datetime.datetime(1995, 1, 1, 0)
    timestamp_list = [start + datetime.timedelta(hours=x) for x in range(len(df))]
    df['timestamps'] = timestamp_list

    # mistake in hours numbering hour 6 = 1248 instances
    #                                 9 = 1246 instances
    # TODO find and correct such that every hour has 1247 instances

    # hourly change in consumption
    diffMain = np.append(np.array(df['mainGrid']), 0) - np.append(0,np.array(df['mainGrid']))
    diffMain = diffMain[0:-1]
    diffMain[0] = 0
    diffOther = np.append(np.array(df['otherGrids']), 0) - np.append(0,np.array(df['otherGrids']))
    diffOther = diffOther[0:-1]
    diffOther[0] = 0

    df['diffMain'] = diffMain
    df['diffOther'] = diffOther

    df_hourDiffMain = pd.DataFrame()
    for hour in np.arange(24):
        hourDiffList = df.loc[df['hourOfDay'] == hour]['diffMain'].values
        print("hour: " + str(hour))
        print("hour diff list len: " + str(np.size(hourDiffList)))
        df_hourDiffMain[str(hour)] = hourDiffList

    df_hourDiffMain.boxplot()
    plt.show()

    print("Hourly differences in consumption")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.boxplot(diffMain)
    ax1.set_title("main Grid")
    ax2.boxplot(diffOther)
    ax2.set_title("other Grids")
    plt.show()

    plt.plot(df.timestamps[1:], diffMain)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.boxplot(df['mainGrid'])
    ax1.set_title('main grid')
    ax2.boxplot(df['otherGrids'])
    ax2.set_title('other grids')
    plt.show()

    figure(figsize=(22, 5), dpi=80, linewidth=5)
    plt.plot(timestamp_list, df['mainGrid'],'r.')
    plt.plot(timestamp_list, df['otherGrids'], 'b.')
    plt.title('Power constumption')
    plt.xlabel('time', fontsize=14)
    plt.ylabel('consumption', fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()


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
