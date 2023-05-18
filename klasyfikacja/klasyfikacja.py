#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:07:51 2019

@author: kamilkurzyca
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
os.getcwd()
#ladujemy df
df=pd.read_csv('/Users/kamilkurzyca/Desktop/klasyfikacja/train_data.csv')

df=df.drop(['Unnamed: 0','PassengerId'] ,axis=1)

df.columns

y=df['Survived'].values
X=df[['Sex','Age','Fare']].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)



clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)  
pred=clf.predict(X_test)

confusion_matrix(y_test,pred)
accuracy_score(y_test,pred)
f1_score(y_test,pred)




fpr, tpr, thresholds =roc_curve(y_test, pred)

