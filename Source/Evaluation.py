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
    digits = 3
    print(f'MSE is : {np.round(mean_squared_error(y_test, y_pred), digits)}')
    print(f'MAE is : {np.round(mean_absolute_error(y_test, y_pred), digits)}')
    print(f'RMSE is : {np.round(np.sqrt(mean_squared_error(y_test, y_pred)), digits)}')
    print(f'R2 is : {np.round(r2_score(y_test, y_pred), digits)}', end='\n\n')


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


def plotActualVsPred(y_test, y_pred, target_feature, t_test=None):
    """
    Plot true test data values vs predicted values

    :param t_test: timestamps of predicted data, if not set index is generated starting from 0
    :param target_feature: predicted feature name as str
    :param y_test: true test data
    :param y_pred: predicted values for the test data
    :return:
    """
    plt.figure(figsize=(13, 8))
    if t_test is not None:
        plt.plot(t_test, y_test)
        plt.plot(t_test, y_pred)
    else:
        plt.plot(list(y_test))
        plt.plot(list(y_pred))
    plt.title("Actual vs Predicted")
    plt.ylabel(target_feature)
    plt.legend(('Actual', 'predicted'))
    plt.show()
