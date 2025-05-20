import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\ky321\OneDrive\Desktop\Data science and Ai\Machine Learning\regression\non linear regression\polynomial regression\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x, y, color ='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("linear  regression model (linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

m=lin_reg.coef_
print(m)

c=lin_reg.intercept_
print(c)

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

#polynomial regression(non linear regression)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x, y, color ='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("polynomial  regression model (non linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

poly_model_pred = lin_reg2.predict(poly_reg.transform([[6.5]]))
poly_model_pred



