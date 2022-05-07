# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:46:10 2022

@author: khali

this file contains some functions that are important for :
    dividing datasets into features and labels
    encoding of categorical data
    missing values computation

"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.api as sm 

#============================================================================

#function to divide datasets into x and y (features and output)

def dataset_dis(csv_data_path):
    
    data = pd.read_csv(csv_data_path)
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y = y.reshape(len(y),1)
    
    return X , y

#============================================================================

#function to onehot encode categorical data
def dummyEncoding( X , column_number ,rem = "passthrough" ):
    CT = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[column_number])],remainder = rem) 
    X = np.array(CT.fit_transform(X))
    return X

#============================================================================

def missing_value_computation(X,range_ ,missing_values_type = np.nan,Strategy = "mean"):
    initial = range_[0]
    final = range_[1]
    imputer = SimpleImputer(missing_values = missing_values_type ,strategy = Strategy )
    imputer.fit(X[:,initial:final])
    X[:,initial:final] = imputer.transform(X[:,initial:final])
    
    return X

#=========================================================================

    
def auto_OLS_model(X,y,SL):
    
    X = np.array(X, dtype=float)
    regressor_OLS = sm.OLS( y,X).fit()
    print(regressor_OLS.summary()) 
    k = X.shape[1]       # k = 6
    if(max(regressor_OLS.pvalues) > SL ):
        for i in range(X.shape[1]):
            k = k - 1
            for j in list(range(k)):
                if(regressor_OLS.pvalues[j] >= max(regressor_OLS.pvalues)):
                    X = np.delete(X,j,axis = 1)
                    X = np.array(X, dtype=float)
                    regressor_OLS = sm.OLS( y,X).fit()
                    print(regressor_OLS.summary())
                    break
    print(regressor_OLS.summary())                
    return X
#==========================================================================