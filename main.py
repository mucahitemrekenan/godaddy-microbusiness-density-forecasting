import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib
import warnings
import math
warnings.filterwarnings('module')
matplotlib.use('QtAgg')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

train['date'] = pd.to_datetime(train['first_day_of_month'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month

train.drop(columns=train.columns.difference(['cfips', 'year', 'month', 
                                             'microbusiness_density']), inplace=True)

x_train = train.drop(columns='microbusiness_density').copy()
y_train = train['microbusiness_density'].copy()

test['date'] = pd.to_datetime(test['first_day_of_month'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month

test.drop(columns=test.columns.difference(['cfips', 'year', 'month']), inplace=True)

# mm_scaler = MinMaxScaler((0,1))
# x_train['cfips'] = mm_scaler.fit_transform(x_train[['cfips']])
# test['cfips'] = mm_scaler.transform(test[['cfips']])

lgb = LGBMRegressor(n_estimators=10000, max_depth=-1)
lgb.fit(x_train, y_train, categorical_feature=['cfips'])
predictions = lgb.predict(test)
predictions_train = lgb.predict(x_train)

plt.plot(pd.Series(y_train), label='actual')
plt.plot(pd.Series(predictions_train), label='prediction')
plt.legend()
plt.show()

sample_sub['microbusiness_density'] = pd.Series(predictions)
sample_sub.to_csv('sample_submission_1_01.csv', index=False)