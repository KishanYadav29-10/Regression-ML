# Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv(r'C:\Users\ky321\OneDrive\Desktop\Data science and Ai\Machine Learning\regression\decision tree reg\emp_sal.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=15, random_state=0)
regressor.fit(x_train, y_train)

# Visualize the results
x_grid = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
plt.scatter(x, y, color='red', label='Actual')
plt.plot(x_grid, regressor.predict(x_grid), color='blue', label='Prediction')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
