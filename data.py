import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv('C:/Users/zeebu/OneDrive/Documents/Python/Regression-Models/cars.csv')

x = df['speed'].values 
y = df['dist'].values

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))
plt.scatter(x,y)
plt.plot(x, mymodel)
plt.savefig("my_plot.png")
plt.show()

