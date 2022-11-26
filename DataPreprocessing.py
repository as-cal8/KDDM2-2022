

header = {"mainGrid", "otherGrids", "hourOfDay"}





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