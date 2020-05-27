import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd 

trainingSet = pd.read_csv('set.csv')

print(trainingSet)

plt.scatter(trainingSet.Area, trainingSet.Price, color= "red")


plt.xlabel("Area")
plt.ylabel("Prices")


lg = linear_model.LinearRegression()
lg.fit(trainingSet[['Area']], trainingSet.Price )

plt.plot(trainingSet.Area, lg.predict(trainingSet[['Area']]), color = 'Green')

plt.show()

print("\n", lg.predict([[2900]] ))