import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import sys
import os
path = os.getcwd()
if path.endswith('LSTM'):
    path = path[:-9]
path = path + str("Source")
sys.path.insert(0, path)
pathData = path + str("\Data")
print(pathData)

def arima():
    df= pd.read_csv('./drive/MyDrive/noise.csv')

    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(df, autolag = 'AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)
    # Since the p-value is small it is stationary

    from pmdarima import auto_arima

    stepwise_fit = auto_arima(df, trace=True,
    suppress_warnings=True)

    print(df.shape)
    train_size = 23927- 2728
    train=df.iloc[:train_size]
    test=df.iloc[23928:]
    print(train.shape,test.shape)

    from statsmodels.tsa.arima.model import ARIMA

    model=ARIMA(train,order=(2,1,4))
    model=model.fit()
    model.summary()

    start= 19141+2786
    end=19141+2786+6000
    true_predictions=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
    true_predictions.plot(legend=True)

    from sklearn.metrics import mean_squared_error
    df_train, df_test = dataPreprocesssing()
    t = df_test.iloc[:, 0].values
    print(len(t))
    add = additiveModel(df_train, t, ignore_plots=True)
    # len(add)
    print(type(add))
    print(type(true_predictions))
    print(add.shape)
    print(true_predictions.shape)
    plt.plot(add)
    plt.show()
    final = []
    test = df_test['mainGrid'].values

    predict = true_predictions.values
    for i in range(0,6000,1):
        final.append(add[i] + predict[i])
    print(len(final))
    plt.plot(final)
    plt.plot(df_test['mainGrid'].values)
    plt.show()
    np.sqrt(mean_squared_error(df_test['mainGrid'].values, final))

    #

    model.save('./drive/MyDrive/model_arima_better.pkl')

if __name__ == '__main__':
    arima()
# load model
