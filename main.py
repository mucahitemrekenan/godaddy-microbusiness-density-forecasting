import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import lightgbm
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
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

preds_test = pd.DataFrame(index=test.index, columns=['preds'])
preds_train = pd.DataFrame(index=x_train.index, columns=['preds'])

# mm_scaler = MinMaxScaler((0,1))
# x_train['cfips'] = mm_scaler.fit_transform(x_train[['cfips']])
# test['cfips'] = mm_scaler.transform(test[['cfips']])

# model = LGBMRegressor(n_estimators=2000, max_depth=4, num_leaves=8)
for county in tqdm(x_train['cfips'].unique()):
    model = LinearRegression()

    index_train = x_train.loc[x_train['cfips'] == county].index
    index_test = test.loc[test['cfips'] == county].index

    model.fit(x_train.loc[index_train], y_train[index_train])
    preds_test.loc[index_test, 'preds'] = model.predict(test.loc[index_test])
    preds_train.loc[index_train, 'preds'] = model.predict(x_train.loc[index_train])

# plt.plot(pd.Series(y_train), label='actual')
# plt.plot(pd.Series(preds_train['preds']), label='prediction')
# plt.legend()
# plt.show()

sample_sub['microbusiness_density'] = pd.Series(preds_test['preds'])
sample_sub['microbusiness_density'] += 0.055
sample_sub.to_csv('submissions/sample_submission_1_22.csv', index=False)