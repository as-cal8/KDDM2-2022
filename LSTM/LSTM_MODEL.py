import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.preprocessing.sequence import TimeseriesGenerator

import sys
import os

path = os.getcwd()
if path.endswith('LSTM'):
    path = path[:-4]
path = path + str("Source")
sys.path.insert(0, path)

from DataPreprocessing import dataPreprocesssing
from AdditiveModel import additiveModel

pathData = path + str("/Data")

TRAIN_MODEL = False #Make it True for Training new models
Testing_noise = False # Evaluating Noise Data for additive model

TRIAN_SIZE = 23928 #80 percent of the dataset
MODELS_NAMES = ['./Mt8','./Mt9','./Mt10','./Mt11','./Mt12','./Mt13','./Mt14']
CSV_FILE = 'LSTM_models_performance'

LOOK_BACK_LIST = [12,12,5,24,24,24,24]
LOOK_BACK = 12 #Used only for traning

if(Testing_noise):
            MODELS_NAMES = ['Mt5_noise','.Mt6_noise','.Mt7_noise','.Mt15_noise','.Mt16_noise','.Mt17_noise']
            LOOK_BACK_LIST = [12,12,12,24,24,5]
            CSV_FILE = 'LSTM_models_noise_performance'


def extracted_features():

    if Testing_noise:
      dataframe= pd.read_csv(pathData+'/noise.csv')
    else:
      dataframe= pd.read_csv(pathData+'/powerSupplyStream.csv', usecols=['mainGrid'])

    dataset = dataframe.values
    dataset = dataset.astype('float64')

    #Split the dataset into train/ validate/ test sets
    if Testing_noise:
      train = dataset[:15312]
      valid = dataset[15312:20098]
      test = dataset[20098:]
    else:
      train = dataset[:19143]
      valid = dataset[19143:23928]
      test = dataset[23928:]

    #normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(train)
    dataset_valid = scaler.fit_transform(valid)
    dataset_test = scaler.fit_transform(test)

    # Generating timeseries dataset
    look_back= LOOK_BACK
    n_features = 1
    generator_train = TimeseriesGenerator(dataset_train, dataset_train, length=look_back, batch_size=1)
    generator_valid = TimeseriesGenerator(dataset_valid, dataset_valid, length=look_back, batch_size=1)

    return generator_train, generator_valid, dataset_train, dataset_test, scaler

def LSTM_Model(generator_train, generator_valid, scaler ):
    #Build Model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)))

    model.add(LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)))


    model.add(LSTM(64, return_sequences=False))


    model.add(Dense(1))

    model.compile(loss='mean_squared_error',optimizer="adam", metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    model.summary()
    early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=100, verbose=0, mode='min')
    save_best = ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_root_mean_squared_error', mode='min')
    model.fit(generator_train, epochs=60, batch_size=32, verbose=True, validation_data =generator_valid , callbacks = [early_stopping,save_best])
    loss_per_epoch = model.history.history['val_loss']

    return model

def evaluate_model(model,dataset_train, dataset_test, scaler,look_back):
    print(model.summary())
    test_predictions = []

    first_eval_batch = dataset_train[-look_back:]
    current_batch = first_eval_batch.reshape((1, look_back, 1))

    for i in range(6000):
        current_prediction = model.predict(current_batch, verbose=0)[0]
        test_predictions.append(current_prediction)
        current_batch = np.append(current_batch[:,1:,:],[[current_prediction]],axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    dataset_test = scaler.inverse_transform(dataset_test)

    # plot baseline and predictions
    plt.figure(figsize=(15,10))
    plt.plot(dataset_test[:1000],label='Actual')
    plt.plot(true_predictions[:1000],label='Predicted')
    plt.xlabel("Time Series in hrs")
    plt.ylabel("Main Grid Values")
    plt.title("Actual vs Prediction")
    plt.legend()
    plt.show()
    plt.plot(dataset_test,label='Real')
    if(Testing_noise):
        plt.title("Noise Values")
    else:
        plt.title("Sensor Values")
    plt.show()

    rmse=np.sqrt(mean_squared_error(dataset_test[:3829],true_predictions[:3829]))#3829
    print(rmse)
    return true_predictions,rmse

def LSTM_Additive_model(predictions):
    df_train, df_test = dataPreprocesssing()
    t = df_test.iloc[:, 0].values
    add = additiveModel(df_train, t, ignore_plots=True)

    plt.plot(add)
    plt.show()
    final = []
    test = df_test['mainGrid'].values

    predict = predictions
    for i in range(0,6000,1):
        final.append(add[i] + predict[i][0])
    plt.figure(figsize=(15,10))
    plt.plot(df_test['mainGrid'].values, label="Actual")
    plt.plot(final, label = 'Predicted')
    plt.show()
    rmse = np.sqrt(mean_squared_error(df_test['mainGrid'].values, final))
    return rmse

def main():

    generator_train, generator_valid,dataset_train, dataset_test, scaler = extracted_features()

    if (TRAIN_MODEL):
        model = LSTM_Model( generator_train, generator_valid, scaler )
    else:
        performance_table = pd.DataFrame(columns = ['Model_Name', 'Test_RMSE', 'Window'])

        for  i in range (len(MODELS_NAMES)):
            model = keras.models.load_model(MODELS_NAMES[i])
            look_back = LOOK_BACK_LIST[i]
            predictions, rmse = evaluate_model(model,dataset_train, dataset_test,scaler,look_back)
            if(Testing_noise):
                rmse_orignal = LSTM_Additive_model(predictions)
                performance_table = performance_table.append({'Model_Name':MODELS_NAMES[i], 'Test_RMSE':rmse_orignal ,'Window': look_back},ignore_index=True)
                print(performance_table)
                performance_table.sort_values(by=['Test_RMSE'],ascending=True)
            else:
                performance_table = performance_table.append({'Model_Name':MODELS_NAMES[i], 'Test_RMSE':rmse,'Window': look_back} ,ignore_index=True)
                print(performance_table)
                performance_table= performance_table.sort_values(by=['Test_RMSE'], ascending=True)
        print(performance_table)
        performance_table.to_csv(CSV_FILE, index=False)



if __name__ == "__main__":
    main()
