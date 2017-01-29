"""
16 Jan 2017
Author - Alex Novikov

The goal of the code is to load in Titanic data from Kaggle,
prepare it and run 3 algorithms - Naive Bayes, SVM and Random Forests.

Variables :
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

"""
import numpy as np
import os
import pandas as pd
from time import time
from sklearn import svm
import matplotlib.pyplot as plt
import math

# check current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# read train data into pandas dataframe
df = pd.read_csv(dir_path + '/data/train.csv', header=0)

# check out types of data in the dataframe:
print df.info()

# drop ultimately useless columns here:
df = df.drop(["Name", "Cabin", "Ticket"], axis=1)

# turn "Sex" character variable into binary
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)


# random distribution = mu + sigma * np.random.rand(dimensions)
age = df['Age'].dropna()
mu, sigma = 30, 14
print round(mu + sigma*np.random.randn())

# replace missing train age values with normally distributed values (mean, standard deviation of Titanic population)
for i in range(0,len(df),1):
    if df["Age"].isnull()[i] == True:
        df.at[i,"Age"] = round(mu + sigma*np.random.randn())

# histogram of age (missing values imputed)
# plt.hist(df2["Fare"])
# plt.show()

# import actual test data
df2 = pd.read_csv(dir_path + '/data/test.csv', header=0)
# turn "Sex" character variable into binary
df2['Gender'] = df2['Sex'].map({'female': 0, 'male': 1}).astype(int)
# drop useless variables
df2 = df2.drop(["Name", "Cabin", "Sex"], axis=1)
'''
# replace zero fare values with mean
for index, row in df2.iterrows():
    if row["Fare"] == 0:
        row["Fare"] = np.mean(df2["Fare"])
'''
df2["Fare"] = df2["Fare"].fillna(df2["Fare"].mean())

# replace missing test age values with normally distributed values (mean, standard deviation
#  of TRAIN Titanic population)
for i in range(0, len(df2), 1):
    if df2["Age"].isnull()[i] == True:
        df2.at[i, "Age"] = round(mu + sigma * np.random.randn())

# define features and labels of the train data

df_features_train = df[["Pclass","Gender","Fare", "Age"]]
features_train = np.array(df_features_train.values)
df_labels_train = df["Survived"]
labels_train= np.array(df_labels_train.values).ravel()
# and features for test data
df_features_test = df2[["Pclass","Gender","Fare", "Age"]]
print df2[["Pclass","Gender","SibSp","Parch","Fare", "Age"]].info()
features_test= np.array(df_features_test.values)



# Try out Random forests

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=8, criterion="gini", oob_score=False)
t = time()
print "training starts now..."
clf = clf.fit(features_train, labels_train)
print "...and  training time is ", round(time()-t, 3), "s"
predictions = clf.predict(features_test)



# output results
result = np.c_[df2.ix[:,0].astype(int), predictions.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv(dir_path + 'titanic_results3.csv', index=False)





# TODO  do some analysis on cabin location... (proximity calculation?)

