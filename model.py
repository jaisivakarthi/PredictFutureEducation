#!/usr/bin/env python
# coding: utf-8

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

#Import Dataset
path = r'data_set.csv'
df = pd.read_csv(path)
print(np.where(pd.isnull(df)))

#Analyze the Dataset
# df.info()
# df.describe()
# df.head()

#Feature selection
x = df.iloc[:, 1:51].values
y = df.iloc[:, -1:].values

#Convert the categorical variable to numerical form
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Split dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x, y, test_size = .2, random_state=0)
#len(x_train), len(x_test)

#Apply models
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
score = logmodel.score(x_train, y_train)
predictions=logmodel.predict(x_test) #predict the result
# print(score, predictions)

#Evaluate the model
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

# Saving model to disk
pickle.dump(logmodel, open('model.pkl','wb'))

# Loading model to compare the results
# Predict the result of salary based on Years of experience
model = pickle.load(open('model.pkl','rb'))
# print(model.predict(x_test))

courses = {0:'Arts', 1:'BA', 2:'Commerce', 3:'Engineering', 4:'Medical' }
result = []
result = [courses[i] for i in model.predict(x_test) if i in courses.keys()]
print(result)