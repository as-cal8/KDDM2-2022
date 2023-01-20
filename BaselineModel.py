from DataPreprocessing import dataPreprocesssing
import numpy as np
from AdditiveModel import accuracy

'''
Naive baseline model, which predicts the test data randomly
in the range of the train data since there is no trend
'''
def baselineModel(df_train, df_test):
    limup = np.max(df_train.mainGrid)
    limlow = np.min(df_train.mainGrid)

    prediction = np.random.uniform(low=limlow, high=limup, size=np.size(df_test.mainGrid))

    accuracy(prediction, df_test.mainGrid)


if __name__ == "__main__":
    df_train, df_test = dataPreprocesssing()
    baselineModel(df_train, df_test)