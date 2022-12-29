# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:25:28 2022

@author: JAMES
"""

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
dataset=pd.read_csv('gtb.csv')
print(dataset)
x=dataset[['nb_places',	'nb_porte',	'automatique']]
y=dataset['prix']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#print(dataset)
print(x_train)
redresseur=LinearRegression()
redresseur.fit(x_train,y_train)
LinearRegression()
y_pred=redresseur.predict(x_test)
df=pd.DataFrame({'notre actuelle valeurs':y_test,'notre valeurs predites':y_pred})
print(df)
joblib.dump(redresseur, 'dangote.pkl')
