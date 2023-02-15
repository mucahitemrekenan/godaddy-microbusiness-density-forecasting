import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor


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

lgb = LGBMRegressor(n_estimators=7000, max_depth=-1)
lgb.fit(x_train, y_train, categorical_feature=['cfips', 'year', 'month'])
predictions = lgb.predict(test)

