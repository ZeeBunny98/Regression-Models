import matplotlib.pyplot as plt
import pandas as pd
import numpy

df = pd.read_csv('C:/Users/zeebu/OneDrive/Documents/Python/Regression-Models/cars.csv')

x = df['speed'].values 
y = df['dist'].values

mymodel = numpy.poly1d(numpy.polyfit(x,y,3))
myline = numpy.linspace(1, 25, 120)

plt.scatter(x,y)
plt.plot(myline, mymodel(myline))
plt.show()