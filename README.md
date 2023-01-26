# Timeseries Prediction

##Task: Given a set of sensor data for multiple streams, e.g. temperature, power consumption. The goal is to predict the future values of these signals, optimally including a confidence range.

Approach: Take the stream of data and remove the last ~10% of the data. Build a prediction algorithm that is able to predict the future values of the streams as accurately as possible and compare against the values you removed.

    Data Set:
    - Powersupply Stream
      Use the power supply stream from the same data source: Stream Data Mining Repository. Here the challenge is to integrate seasonality into the analysis.
      https://www.cse.fau.edu/~xqzhu/stream.html

Advanced: Investigate your prediction algorithm and try to determine under which (controlled) circumstances it makes correct predictions.


## Prerequisites

The Requirements to reproduce these results are as follows:

|Package            |Version |
|-------------------|--------|
| Python            | 3.8.16 |
| Numpy             | 1.23.3 |
| Scipy             | 1.9.3  |
|pmdarima           | 2.0.2  |
|statsmodels        |        |
|tensorflow         | 2.10.0 |
|Keras              | 2.10   |

Correct Functionality and comparable Results cannot be guaranteed on different Versions.

## To Train LSTM Model
Preferred to run this code on linux or mac os.
Enter the and set flag TRAIN_MODEL = true
Choose a Window size by setting the LOOK_BACK variable

## To Evaluate LSTM Model
Preferred to run this code on linux or mac os.
Enter the LSTM_Model.py file and set flag TRAIN_MODEL = False
If testing noise data( from additive model), set the testing_noise = True otherwise set it to False

## To Train ARIMA Model
Preferred to run this code on linux or mac os
Enter arima_model.py
