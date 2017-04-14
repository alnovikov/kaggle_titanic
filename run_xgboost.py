import numpy as np
import os
import pandas as pd
import xgboost as xgb
from time import time

# read train and data into pandas dataframe
df = pd.read_csv('/data/train.csv', header=0)
df2 = pd.read_csv('/data/test.csv', header=0)

# turn "Sex" character variable into binary
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df2['Gender'] = df2['Sex'].map({'female': 0, 'male': 1}).astype(int)

#fill NAs in Age with random values drawn from normal distribution
mu, sigma = np.mean(df['Age'].dropna()), np.std(df['Age'].dropna())
df["Age"] = df["Age"].fillna(round(mu + sigma*np.random.randn()))

# define features and labels of the train data
df_features_train = df[["Pclass","Gender","Fare", "Age"]]
df_labels_train = df["Survived"]

features_train = np.array(df_features_train.values)
labels_train= np.array(df_labels_train.values).ravel()

# and features for test data
df_features_test = df2[["Pclass","Gender","Fare", "Age"]]
features_test= np.array(df_features_test.values)

#xgboosting
X  = xgb.DMatrix(features_train)
Y = xgb.DMatrix(features_test)

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, X, num_round)
# make prediction
preds = bst.predict(dtest)
