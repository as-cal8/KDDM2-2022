from Source.DataPreprocessing import dataPreprocesssing
from xgboost import XGBRegressor
import lightgbm as lgb
from Evaluation import *

# Sources used: https://www.kaggle.com/code/enesdilsiz/time-series-forecasting-with-lightgbm/data
#               https://michael-fuchs-python.netlify.app/2020/11/10/time-series-analysis-xgboost-for-univariate-time-series/

def createTimeFeatures(df, target_feature):
    """
    Creates time series features from timestamps feature

    :param df: dataframe containing "timestamps" feature, which represents date + time
    :param target_feature: name of target feature as str
    :return: dataframe containing time features
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
    if target_feature:
        y = df[target_feature]
        return X, y
    return X

df_train, df_test = dataPreprocesssing()
df_train.rename(columns={'Unnamed: 0':'hour'}, inplace=True)
df_test.rename(columns={'Unnamed: 0':'hour'}, inplace=True)

tsplot(df_train['mainGrid'])

trainX, trainY = createTimeFeatures(df_train, 'mainGrid')
testX, testY = createTimeFeatures(df_test, 'mainGrid')

'''
XGBoost
'''
xgb = XGBRegressor(objective= 'reg:linear', n_estimators=1000)
xgb.fit(trainX, trainY, verbose=False)
predicted_results = xgb.predict(testX)
evaluationMetrics(testY, predicted_results)

plotActualVsPred(testY, predicted_results, 'mainGrid')


'''
LightGBM
'''
lgbm = lgb.LGBMRegressor(n_estimators=1000, objective="regression")
lgbm.fit(trainX, trainY)
predicted_results = lgbm.predict(testX)
evaluationMetrics(testY, predicted_results)

plotActualVsPred(testY, predicted_results, 'mainGrid')