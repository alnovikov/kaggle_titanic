import numpy as np
import os
import pandas as pd
from time import time
import matplotlib.pyplot as plt

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
print round(mu + sigma * np.random.randn())

# replace missing train age values with normally distributed values (mean, standard deviation of Titanic population)
for i in range(0, len(df), 1):
    if df["Age"].isnull()[i] == True:
        df.at[i, "Age"] = round(mu + sigma * np.random.randn())

# or could have replaced with this line of code...
df["Age"] = df["Age"].fillna(round(mu + sigma * np.random.randn()))

# histogram of age (missing values imputed) to check out how the distribution looks like
plt.hist(df["Age"])
plt.show()

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
df_features_train = df[["Pclass", "Gender", "Fare", "Age"]]
features_train = np.array(df_features_train.values)
df_labels_train = df["Survived"]
labels_train = np.array(df_labels_train.values).ravel()
# and features for test data
df_features_test = df2[["Pclass", "Gender", "Fare", "Age"]]
print df2[["Pclass", "Gender", "SibSp", "Parch", "Fare", "Age"]].info()
features_test = np.array(df_features_test.values)

# Try out Neural Network
# First, pre-process: scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(features_train)
X_train = scaler.transform(features_train)
X_test = scaler.transform(features_test)

# Train classifier:
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
t = time()
print "training starts now..."
clf = clf.fit(X_train, labels_train)
print "...and  training time is ", round(time() - t, 3), "s"
predictions = clf.predict(X_test)

# output results
result = np.c_[df2.ix[:, 0].astype(int), predictions.astype(int)]
df_result = pd.DataFrame(result[:, 0:2], columns=['PassengerId', 'Survived'])
# and write them to disk
df_result.to_csv(dir_path + 'titanic_results_NN.csv', index=False)
