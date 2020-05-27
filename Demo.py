import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

x = [1,2,3,4]
y = [1,2,3,4]

X = [ [1], [2], [3],[4]]        # we build this array of X because fit() method of Linear Regression takes 2d array as first argument and 1d array as second argument

lg = linear_model.LinearRegression()
lg.fit(X,y)

y_cal = lg.predict([[10]])
print(y_cal)    # it will print [10.] as output and that is correct

