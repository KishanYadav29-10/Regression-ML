import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\ky321\OneDrive\Desktop\Data science and Ai\Machine Learning\regression\non linear regression\polynomial regression\emp_sal.csv')

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

# SVR Model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree =5,gamma='scale')

svr_regressor.fit(x,y)

#predicting the new result thorugh SVR
svr_pred = svr_regressor.predict([[6.5]])
print(svr_pred)

# Visualising the SVR results
plt.scatter(x,y,color ='red')
plt.plot(x,svr_regressor.predict(x),color ='blue')
plt.title('Truthh or Bluff (SVR)')
plt.xlabel('postion level')
plt.ylabel('Salary')
plt.show()

#KNN regressor model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model= KNeighborsRegressor(n_neighbors=5,weights='distance',p=2)
knn_reg_model.fit(x,y)

#predicting the new result thorugh KNN regressor
knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# Visualising the KNN regressor results
x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, knn_reg_model.predict(x_grid), color='green')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()