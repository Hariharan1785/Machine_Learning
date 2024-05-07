
#Linear Regression Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# we will calculate r_square here, calculate other regression metric by yourself
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

link = "https://raw.githubusercontent.com/Hariharan1785/Machine_Learning/main/Linear%20Regression_sample.csv"
# Above Link which is read from GitHub Repository 
link = "C://Users//USER//OneDrive//Top_Mentor//Classes//DataSets//MachineLearning-master/Linear Regression_sample.csv"
dataset = pd.read_csv(link)
print("Data Retrived from GIT Hub Repository", dataset)

#plt.scatter(dataset['R&D Spend'], dataset['Profit'])
#plt.show()
plt.scatter(dataset['X'], dataset['Y'])
plt.show()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Except Last Column", X)
print("Last Column \n", y)



# Split Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
print("X Train Value \n", X_train)
print("y_Train Value \n", y_train)
print("X_Test Value \n", X_test)
print("Y_Test Value \n", y_test)



# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
print("After scaling X_train data: \n", X_train)
print("After scaling X_test data: \n", X_test)
# OLS Model Predicting for the P_value


from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()

Regressor.fit(X_train,y_train)

print("Intercept",Regressor.intercept_)
print("Slope",Regressor.coef_)


y_pred = Regressor.predict(X_test)
outcome_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(outcome_df)

# MAE
from sklearn import metrics
import numpy as np
mae = metrics.mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error", mae)

# MSE
mse = metrics.mean_squared_error(y_pred, y_test)
print("Mean Squared Error", mse)

# RMSE
rmse = np.sqrt(mse)
print("Root Mean Square", rmse)

# Rsquare
R_Square = metrics.r2_score(y_pred, y_test)
print("R-Square", R_Square)


