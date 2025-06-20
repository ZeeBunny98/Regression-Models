import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pydataset import data

df = pd.read_csv('C:/Users/zeebu/OneDrive/Documents/Python/Regression-Models/cars.csv')

x = df['speed'].values.reshape(-1, 1) 
y = df['dist'].values

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='speed', y='dist', data=df, color='blue', s=60)

plt.plot(df['speed'], y_pred, color='red', label='sklearn Line')

plt.grid(True)
plt.show()