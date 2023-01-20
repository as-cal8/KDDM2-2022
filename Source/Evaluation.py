import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *
import statsmodels.api as sm
import statsmodels.tsa.api as smt

def evaluationMetrics(y_test, y_pred):
    """
    Calculate the following evaluation metrics:
        - MSE
        - MAE
        - RMSE
        - R^2

    :param y_test: true values of the test data
    :param y_pred: predicted values for the test data
    :return: MSE, MAE, RMSE, and R^2
    """
    print('Evaluation metric results: ')
    print(f'MSE is : {mean_squared_error(y_test, y_pred)}')
    print(f'MAE is : {mean_absolute_error(y_test, y_pred)}')
    print(f'RMSE is : {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R2 is : {r2_score(y_test, y_pred)}', end='\n\n')

# Source: https://towardsdatascience.com/multi-step-time-series-forecasting-with-arima-lightgbm-and-prophet-cc9e3f95dfb0
def tsplot(y, lags=None, figsize=(12, 7)):
    """
    Plot time series, its ACF and PACF, calculate Dickey–Fuller test

    :param y: timeseries
    :param lags: how many lags to include in ACF, PACF calculation
    :param figsize: figure size default (12, 7)
    :return:
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, method='ywm')
    plt.tight_layout()

def plotActualVsPred(y_test, predicted_results):
    """
    Plot true test data values vs predicted values

    :param y_test: true test data
    :param predicted_results: predicted values for the test data
    :return:
    """
    plt.figure(figsize=(13, 8))
    plt.plot(list(y_test))
    plt.plot(list(predicted_results))
    plt.title("Actual vs Predicted")
    plt.ylabel("mainGrid")
    plt.legend(('Actual', 'predicted'))
    plt.show()

