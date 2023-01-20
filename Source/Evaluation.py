import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *
import statsmodels.api as sm
import statsmodels.tsa.api as smt

def timeseries_evaluation_metrics_func(y_true, y_pred):
    '''
    Calculate the following evaluation metrics:
        - MSE
        - MAE
        - RMSE
        - R²

    Args:
        y_true (float64): Y values for the dependent variable (test part), numpy array of floats
        y_pred (float64): Predicted values for the dependen variable (test parrt), numpy array of floats

    Returns:
        MSE, MAE, RMSE, and R²
    '''
    print('Evaluation metric results: ')
    print(f'MSE is : {mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(mean_squared_error(y_true, y_pred))}')
    print(f'R2 is : {r2_score(y_true, y_pred)}', end='\n\n')

def tsplot(y, lags=None, figsize=(12, 7)):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
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

def plotActualVsPred(testY, predicted_results):
    plt.figure(figsize=(13, 8))
    plt.plot(list(testY))
    plt.plot(list(predicted_results))
    plt.title("Actual vs Predicted")
    plt.ylabel("mainGrid")
    plt.legend(('Actual', 'predicted'))
    plt.show()

