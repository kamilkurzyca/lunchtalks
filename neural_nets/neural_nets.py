#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:37:50 2019

@author: kamilkurzyca
"""

import keras 
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
os.getcwd()
#os.chdir(os.getcwd()+'/Desktop/neural_nets')


df = pd.read_json(os.getcwd()+'/train.json')

#print(df.head(), '\n')
def preproc(df):
    target_variables = np.unique(df['cuisine'], return_counts=False)
    #print('the number of target variables (cuisines) is: {}'.format(len(target_variables)), '\n')
    
    
    # Find the number of all distinct ingredients available
    set_of_all_ingredients = set()
    for list_of_ingredients in df['ingredients']:
        for ingredient in list_of_ingredients:
            set_of_all_ingredients |= set([ingredient])
    
    list_of_all_ingredients = list(set_of_all_ingredients)
    
    
    df['ingredients_without_spaces'] = df['ingredients'].apply(','.join)
    
    vect = TfidfVectorizer(binary=True).fit(df['ingredients_without_spaces'].values)
    X = vect.transform(df['ingredients_without_spaces'].values)
    X = X.astype('float')
    
    
    encoder = OneHotEncoder()
    y = encoder.fit_transform(df.cuisine.values.reshape(-1,1))
    
    X=X[0:3000]
    y=y[0:3000]
    return(y,X)
y,X=preproc(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_train=X_train.toarray()
X_test=X_test.toarray()


model = Sequential()
model.add(Dense(10, input_dim=X_train.T.shape[0], activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(X_train[0:1000], y_train[0:1000], epochs=100, batch_size=100,validation_data=(X_test[0:100], y_test[0:100]),  verbose=1, shuffle=True)





