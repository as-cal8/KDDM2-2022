from Source.DataPreprocessing import dataPreprocesssing
from xgboost import XGBRegressor
import lightgbm as lgb
from Evaluation import *

def create_features(df, target_variable):
    """
    Creates time series features from datetime index

    Args:
        df (float64): Values to be added to the model incl. corresponding datetime
                      , numpy array of floats
        target_variable (string): Name of the target variable within df

    Returns:
        X (int): Extracted values from datetime index, dataframe
        y (int): Values of target variable, numpy array of integers
    """
    df['date'] = df.index
    df['hour'] = df['timestamps'].dt.hour
    df['dayofweek'] = df['timestamps'].dt.dayofweek
    df['quarter'] = df['timestamps'].dt.quarter
    df['month'] = df['timestamps'].dt.month
    df['year'] = df['timestamps'].dt.year
    df['dayofyear'] = df['timestamps'].dt.dayofyear
    df['dayofmonth'] = df['timestamps'].dt.day
    df['weekofyear'] = df.timestamps.dt.isocalendar().week.astype('int64')

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if target_variable:
        y = df[target_variable]
        return X, y
    return X

df_train, df_test = dataPreprocesssing()
df_train.rename(columns={'Unnamed: 0':'hour'}, inplace=True)
df_test.rename(columns={'Unnamed: 0':'hour'}, inplace=True)

tsplot(df_train['mainGrid'])

trainX, trainY = create_features(df_train, 'mainGrid')
testX, testY = create_features(df_test, 'mainGrid')

'''
XGBoost
'''
xgb = XGBRegressor(objective= 'reg:linear', n_estimators=1000)
xgb.fit(trainX, trainY, verbose=False)
predicted_results = xgb.predict(testX)
timeseries_evaluation_metrics_func(testY, predicted_results)

plotActualVsPred(testY, predicted_results)


'''
LightGBM
'''
lgbm = lgb.LGBMRegressor(n_estimators=1000, objective="regression")
lgbm.fit(trainX, trainY)
predicted_results = lgbm.predict(testX)
timeseries_evaluation_metrics_func(testY, predicted_results)

plotActualVsPred(testY, predicted_results)