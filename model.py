# -*- coding: utf-8 -*-

import itertools

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('Agg')
dataset = pd.read_csv('https://www.dropbox.com/s/cew7nwsn6erqj0m/pima_diabetes.csv?dl=1')

from sklearn.model_selection import train_test_split
X = dataset
y = dataset['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Filling nulls and zeros
X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==0, 'Glucose'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==1, 'Glucose'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==0, 'Glucose'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==1, 'Glucose'].median()


X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==0, 'BloodPressure'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==1, 'BloodPressure'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==0, 'BloodPressure'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==1, 'BloodPressure'].median()

X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==0, 'SkinThickness'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==1, 'SkinThickness'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==0, 'SkinThickness'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==1, 'SkinThickness'].median()


X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==0, 'Insulin'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==1, 'Insulin'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==0, 'Insulin'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==1, 'Insulin'].median()

X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==0, 'BMI'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==1, 'BMI'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==0, 'BMI'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==1, 'BMI'].median()


X_train = X_train.drop("Outcome", axis=1)
X_test = X_test.drop("Outcome", axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.mean(axis=0)


# Performing oversampling using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# Random forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

# Evaluation
from sklearn.metrics import precision_score, recall_score, confusion_matrix

confusion_matrix_result = confusion_matrix(y_test,y_predicted)
confusion_matrix_result

labels = ['No diabetes','Diabetes']
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix_result,annot=True,cmap='Reds',fmt='.0f',xticklabels=labels,yticklabels=labels)
plt.title('Diabetes Detection')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.show()

recall = recall_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted)

print("Precision = {} \n Recall = {}".format(precision, recall))


with open("model.pkl", 'wb') as file:
    pickle.dump(model, file)

with open("standard_scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)

