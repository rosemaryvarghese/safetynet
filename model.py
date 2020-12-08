import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
data=pd.read_csv(r"newdata.csv")

data['murder and attempt']=data.iloc[:, -30:-27].sum(axis=1)
data['Rape']=data.iloc[:, -28:-25].sum(axis=1)
data.drop(data.columns[[3,4,5,6,7,8]], axis = 1, inplace = True)
data['Kidnapping and Abduction']=data.iloc[:, -26:-23].sum(axis=1)
data.drop(data.columns[[3,4,5]], axis = 1, inplace = True)
data.drop(data.columns[[4,10,11,12,13,14]], axis = 1, inplace = True)
data['Dacoity and various theft']=data.iloc[:, -18:-12].sum(axis=1)
data.drop(data.columns[[3,4,5,6,7,8]], axis = 1, inplace = True)
data.drop(data.columns[[3,7,8,10,11]], axis = 1, inplace = True)
data['violence against women']=data.iloc[:, -8:-5].sum(axis=1)
data.drop(data.columns[[3,4,5]], axis = 1, inplace = True)
data.drop(data.columns[[3]], axis=1, inplace = True)
data.rename(columns = {'STATE/UT':'STATE'}, inplace = True)
data.head()

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

data['STATE']= label_encoder.fit_transform(data['STATE'])
data['DISTRICT']= label_encoder.fit_transform(data['DISTRICT']) 
print(data.head())

X = data.iloc[:,[0,1,2]].values
Y = data.iloc[:,[3,4,5,6,7]].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=50)

rfc = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model = MultiOutputClassifier(rfc, n_jobs=-1)

model.fit(X_train, Y_train)

pickle.dump(model,open('model.pkl','wb'))
