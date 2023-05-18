#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:27:38 2019

@author: kamilkurzyca
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
os.getcwd()
#ladujemy df
df=pd.read_csv('/Users/kamilkurzyca/Desktop/clustering/weight-height.csv')
#dzielimy na podproby w celu wykreslenia
df_male=df[(df.Gender =='Male')][['Height','Weight']].values
df_female=df[(df.Gender =='Female')][['Height','Weight']].values
#rysujemy wykres
plt.figure(figsize=(15,15))
plt.plot(df_male.T[0],df_male.T[1],'ro',markersize=2)
plt.plot(df_female.T[0],df_female.T[1],'ko',markersize=2)
plt.show()
#przygotowanie probki do klastrowania
X=df[['Height','Weight']].values
#konstruktor klasy
gmm=GaussianMixture(n_components=2, covariance_type='full')
#uczenie modelu
pred=gmm.fit_predict(X)
#funkcja ktora dzieli wg klucza
def splitter(X,pred):
    chopy=[]
    baby=[]
    for index,s in enumerate(pred):
        if s==1:
            chopy.append(X[index])
        else:
            baby.append(X[index])
    return(np.array(chopy),np.array(baby))
    
#tworzymy wykres
male,female=splitter(X,pred)
plt.figure(figsize=(15,15))
plt.plot(male.T[0],male.T[1],'ro',markersize=2)
plt.plot(female.T[0],female.T[1],'ko',markersize=2)
plt.show()


print(silhouette_score(X,pred))
del(X,df,female,male,pred,df_female,df_male)
#####################################################################################
df=pd.read_csv('/Users/kamilkurzyca/Desktop/clustering/Mall_Customers.csv')
df=df.drop(['CustomerID'],axis=1)
df['Gender']=np.where(df['Gender']=='Female',1,0)
X=df.values

X_pca=StandardScaler().fit_transform(X)

gmm=GaussianMixture(n_components=3, covariance_type='full')
#uczenie modelu
pred=gmm.fit_predict(X)
gmm.get_params()

pca = PCA(n_components=2)
X_pca=pca.fit_transform(X_pca)  
for index,s in enumerate(pred):
    if s==2:
        print(X[index])
cluster_1,cluster_2=splitter(X,pred)

plt.figure(figsize=(15,15))
plt.plot(cluster_1.T[0],cluster_1.T[1],'ro',markersize=6)
plt.plot(cluster_2.T[0],cluster_2.T[1],'bo',markersize=6)
plt.show()

plt.figure(figsize=(15,15))
plt.scatter(X_pca[:,0], X_pca[:,1], c=pred)

from sklearn.cluster import DBSCAN
pred= DBSCAN(eps=2, min_samples=4).fit_predict(StandardScaler().fit_transform(X))
plt.figure(figsize=(15,15))
plt.scatter(X_pca[:,0], X_pca[:,1], c=pred)
