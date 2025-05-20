import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
dataset = pd.read_csv(r"C:\Users\ky321\OneDrive\Desktop\Data science and Ai\Machine Learning\multiple linear regression\Investment.csv")

# Divide the data into independent and dependent variables
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

# Convert categorical datatype to numerical datatype
x = pd.get_dummies(x, dtype=int)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

# Train the model using Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Make predictions
y_pred = regressor.predict(x_test)

# Display coefficients and intercept
m_slope = regressor.coef_
c_inter = regressor.intercept_
print(m_slope)
print(c_inter)

# Add a column of ones for the intercept term
x = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1)

# Backward Elimination Function
def backward_elimination(x, y, sl=0.05):
    num_vars = len(x[0])
    temp = np.array(x)
    for i in range(num_vars, 0, -1):
        regressor_OLS = sm.OLS(endog=y, exog=temp).fit()
        max_p_value = max(regressor_OLS.pvalues)
        if max_p_value > sl:
            max_p_index = np.argmax(regressor_OLS.pvalues)
            temp = np.delete(temp, max_p_index, axis=1)
        else:
            break
    regressor_OLS.summary()
    return temp, regressor_OLS

# Perform Backward Elimination
x_opt, regressor_OLS = backward_elimination(x, y)

# Display final model summary
print(regressor_OLS.summary())
