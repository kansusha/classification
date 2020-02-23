#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("exp_vs_sal.xlsx")
data.plot(figsize=(7,7))
data.plot.scatter(figsize=(7,7),x = 0, y = 1)

#creating traing test and test set data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)
#Classifying the predictor(X) and target(Y)
X_train = train_set.iloc[:,0].values.reshape(-1, 1)
y_train = train_set.iloc[:,-1].values
X_test = test_set.iloc[:,0].values.reshape(-1, 1)
y_test = test_set.iloc[:,-1].values
#Initializing the KNN Regressor and fitting training data
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2)
regressor.fit(X_train, y_train)
#Predicting Salaries for test set
#Now that we have trained our KNN, we will predict the salaries for the experiences in the test set.
y_pred = regressor.predict(X_test)
#Let’s write the predicted salary back to the test_set so that we can compare.
test_set['Predicted_Salary'] = y_pred
#Visualizing the Predictions vs Actual Observations
plt.figure(figsize = (5,5))
plt.title('Actual vs Predicted Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.scatter(list(test_set["Experience"]),list(test_set["Salary"]))
plt.scatter(list(test_set["Experience"]),list(test_set["Predicted_Salary"]))
plt.show()
