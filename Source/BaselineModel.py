from DataPreprocessing import dataPreprocesssing
from Evaluation import *

'''
Naive baseline model, which predicts the test data randomly
in the range of the train data since there is no trend
'''
def baselineRand(df_train, df_test):
    limup = np.max(df_train.mainGrid)
    limlow = np.min(df_train.mainGrid)

    prediction = np.random.uniform(low=limlow, high=limup, size=np.size(df_test.mainGrid))
    print("Random baseline model:")
    timeseries_evaluation_metrics_func(df_test.mainGrid, prediction)

def baselineMean(df_train, df_test):
    mean = np.mean(df_train.mainGrid)

    prediction = np.ones_like(pd.Series.to_numpy(df_test.mainGrid)) * mean
    print("Mean baseline model:")
    timeseries_evaluation_metrics_func(df_test.mainGrid, prediction)


if __name__ == "__main__":
    df_train, df_test = dataPreprocesssing()
    baselineRand(df_train, df_test)
    baselineMean(df_train, df_test)