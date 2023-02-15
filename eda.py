
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')
census = pd.read_csv('census_starter.csv')

# --analyzing train--
train.columns
# unique value counts of all columns
train.apply(lambda x: len(pd.unique(x).tolist()))
train.isnull().sum()
train['first_day_of_month'].max()
train['first_day_of_month'].min()
train.head()
grouped_cfips = train.groupby(by='cfips').get_group(1001)

# cfips counts differs from county counts , we controlled which county names used for different states
temp = train.groupby(by=['county', 'state']).size().reset_index()

# --analyzing test--
test.columns
# unique value counts of all columns
test.apply(lambda x: len(pd.unique(x).tolist()))
test.isnull().sum()
test['first_day_of_month'].max()
test['first_day_of_month'].min()
test.head()

# --analyzing census--
census.columns
