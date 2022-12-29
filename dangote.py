# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:59:47 2022

@author: JAMES
"""

import pandas
import joblib
model=joblib.load('dangote.pkl')
test=[[5,4,0]]
y_predict=model.predict(test)
y_predict
print(y_predict)