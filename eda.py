
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')
census = pd.read_csv('census_starter.csv')

print(train.columns)
train.apply(lambda x: len(pd.unique(x).tolist()))
train.isnull().sum()
test.isnull().sum()
uniques = train['cfips'].unique()
print(test.columns)
train['first_day_of_month'].max()
test['first_day_of_month'].max()
print(census.columns)

