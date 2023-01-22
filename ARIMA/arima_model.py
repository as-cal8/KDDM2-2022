import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import sys
import os
path = os.getcwd()
if path.endswith('ARIMA'):
    path = path[:-5]
path = path + str("Source")
sys.path.insert(0, path)
pathData = path + str("/Data")
from DataPreprocessing import dataPreprocesssing
from AdditiveModel import additiveModel

def arima():
    df= pd.read_csv(pathData+'/noise.csv')
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

    # Auto Fit the ARIMA Mode and Search the best parameters with the lowest AIC score
    stepwise_fit = auto_arima(df, trace=True,
    suppress_warnings=True)

    train_size = 23927- 2728
    train=df.iloc[:train_size]
    test=df.iloc[23928:]


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
    add = additiveModel(df_train, t, ignore_plots=True)

    plt.plot(add)
    plt.show()
    final = []
    test = df_test['mainGrid'].values

    predict = true_predictions.values
    for i in range(0,6000,1):
        final.append(add[i] + predict[i])
    plt.plot(df_test['mainGrid'].values, label="Actual")
    plt.plot(final,label='Predicted')
    plt.show()
    np.sqrt(mean_squared_error(df_test['mainGrid'].values, final))

    model.save('model_arima.pkl')

if __name__ == '__main__':
    arima()
