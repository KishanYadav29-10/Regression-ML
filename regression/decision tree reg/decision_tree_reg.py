# decision tree regression

# importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 
dataset = pd.read_csv(r'C:\Users\ky321\OneDrive\Desktop\Data science and Ai\Machine Learning\regression\decision tree reg\emp_sal.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# spliting the dataset into Training and test set
from sklearn.model_selection import train_test_split

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='friedman_mse',splitter='random')
regressor.fit(x_train,y_train)
reg = DecisionTreeRegressor(criterion='friedman_mse',splitter='random')
reg.fit(x,y)

# Predict
y_pred = regressor.predict(x_test)

# visualization without grid
# Note:
# Decision tree regression is not a continuous model.
# It splits the feature space into intervals based on information gain.
# For each interval, it predicts the average of the dependent variable (e.g., salary).
# This results in a stepwise (non-linear) prediction curve.
# High-resolution input (x_grid) helps visualize these steps clearly.

plt.scatter(x, y, color = 'red')
plt.plot(x,reg.predict(x), color = 'blue')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

# Visualize the decision tree regression (with high-resolution grid)
x_grid = np.arange(min(x), max(x), 0.01)  # 0.01 step size for smooth steps
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red')  # Actual data
plt.plot(x_grid, regressor.predict(x_grid), color='blue')  # Predictions
plt.title('Truth or Bluff (Decision Tree Regression with grid)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()