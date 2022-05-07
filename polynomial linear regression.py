# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:10:30 2022

@author: khali
"""

import functions as fn 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt 
#========================================================

X , y = fn.dataset_dis("Position_Salaries.csv")
X = X[:,1:2]
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.predict(np.array([6.5]).reshape(1,-1)))
#========================================================

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y ,c = 'black')
plt.plot(X,lin_reg.predict(X),color = "red")

x_grid = np.arange(min(X),max(X)+0.1,0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = "blue")

plt.title("position vs salary")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

#prediction using simple linear regressor model 

print(lin_reg.predict(np.array([6.5]).reshape(1,-1)))

#prediction using polynomial linear regressor model 


print(lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1))))


