
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
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
# from ../../AdditiveModel import additiveModel
#from ../../DataPreprocessing import dataPreprocesssing
TRIAN_SIZE = 23928 #80 percent of the dataset


def extracted_features():

    dataframe= pd.read_csv('noise.csv')
    dataset = dataframe.values
    dataset = dataset.astype('float64')

    print(len(dataframe)-3829)
    train_size = TRIAN_SIZE-3829
    test_size = len(dataset) - train_size

    train = dataset[:15312]
    valid = dataset[15312:20098]
    test = dataset[20098:]

    #normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(train)
    dataset_valid = scaler.fit_transform(valid)
    dataset_test = scaler.fit_transform(test)

    # # split into train and test sets
    look_back= LOOK_BACK
    n_features = 1
    generator_train = TimeseriesGenerator(dataset_train, dataset_train, length=look_back, batch_size=1)
    generator_valid = TimeseriesGenerator(dataset_valid, dataset_valid, length=look_back, batch_size=1)

    return generator_train, generator_valid, dataset_train, dataset_test, scaler

def  LSTM_Model(generator_train, generator_valid, scaler ):


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
    plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    plt.show()
    plt.plot(model.history.history['val_root_mean_squared_error'])
    plt.show()

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
    print(len(t))
    add = additiveModel(df_train, t, ignore_plots=True)
    # len(add)
    print(type(add))
    print(type(predictions))
    print(add.shape)
    print(predictions.shape)
    plt.plot(add)
    plt.show()
    final = []
    test = df_test['mainGrid'].values

    predict = predictions
    for i in range(0,6000,1):
        final.append(add[i] + predict[i][0])
    print(len(final))
    plt.plot(final)
    plt.plot(df_test['mainGrid'].values)
    plt.show()
    rmse = np.sqrt(mean_squared_error(df_test['mainGrid'].values, final))
    return rmse



Path='./drive/MyDrive/KDDM2_Models/'
TRAIN_MODEL = False
TRIAN_SIZE = 23928 #80 percent of the dataset
MODELS_NAMES = ['./Mt9','./Mt10']#,'./Mt11','./Mt12','./Mt13','./Mt14']
CSV_FILE = 'LSTM_models_performance'

MODELS_NAMES_NOISE = [Path+'./Mt16_noise',Path+'./Mt17_noise']#,'./Mt11','./Mt12','./Mt13','./Mt14']
LOOK_BACK_LIST = [12,5]#,24,24,24,24]
LOOK_BACK_NOISE_LIST = [24,5]#,24,24,24,24]
LOOK_BACK = 12
Testing_noise = False
if(Testing_noise):
            MODELS_NAMES = ['./drive/MyDrive/KDDM2_Models/Mt16_noise','./drive/MyDrive/KDDM2_Models/Mt17_noise']
            LOOK_BACK_LIST = [24,5]#,24,24,24,24]
            CSV_FILE = 'LSTM_models_noise_performance'
#mt5_noise window 24
#mt6_noise window 24
#mt7_noise window 24
#mt15_noise window 24
#mt16_noise window 24
#mt17_noise window 5

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
                performance_table.sort_values(by=['Test_RMSE'],inplace=True)
            else:
                performance_table = performance_table.append({'Model_Name':MODELS_NAMES[i], 'Test_RMSE':rmse,'Window': look_back} ,ignore_index=True)
                print(performance_table)
                performance_table= performance_table.sort_values(by=['Test_RMSE'], ascending=True)
        print(performance_table)
        performance_table.to_csv(CSV_FILE, index=False)

if __name__ == "__main__":
    main()
