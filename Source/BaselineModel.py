from DataPreprocessing import dataPreprocesssing
from Evaluation import *

def baselineRand(df_train, df_test):
    """
    Baseline model predicting test values random in range of the train data.
    Prints results of metrics.

    :param df_train: training dataframe, returned by dataPreprocesssing()
    :param df_test: test dataframe, returned by dataPreprocesssing()
    :return:
    """
    limup = np.max(df_train.mainGrid)
    limlow = np.min(df_train.mainGrid)

    prediction = np.random.uniform(low=limlow, high=limup, size=np.size(df_test.mainGrid))
    print("Random baseline model:")
    evaluationMetrics(df_test.mainGrid, prediction)

def baselineMean(df_train, df_test):
    """
    Baseline model predicted values are just the mean of the training data.
    Prints results of metrics.

    :param df_train: training dataframe, returned by dataPreprocesssing()
    :param df_test: test dataframe, returned by dataPreprocesssing()
    :return:
    """
    mean = np.mean(df_train.mainGrid)

    prediction = np.ones_like(pd.Series.to_numpy(df_test.mainGrid)) * mean
    print("Mean baseline model:")
    evaluationMetrics(df_test.mainGrid, prediction)


if __name__ == "__main__":
    df_train, df_test = dataPreprocesssing()
    baselineRand(df_train, df_test)
    baselineMean(df_train, df_test)