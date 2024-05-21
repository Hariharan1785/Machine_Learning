import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

link = "C://Users//USER//OneDrive//Top_Mentor//Classes//DataSets//MachineLearning-master//Titanic-Dataset.csv"
dataset = pd.read_csv(link)
print("Data Retrieved", dataset)
print(dataset.head())
print(dataset.tail())

plt.scatter(dataset['Age'], dataset['Fare'])
plt.show()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_plt = dataset.iloc[:,2:-1].values
print(y)


# Handling missing values
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 4:])
X[:, 4:] = imputer.transform(X[:, 4:])
print(X)


# handling categorical data converting  text to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lc = LabelEncoder()
X[:, 3]= lc.fit_transform(X[:, 3])
print("################# 3rd Column Transfer using Label Encoder #################### \n", lc)
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
print("################Column Transfer Converted 0s and 1s ################################ \n", ct)
X = ct.fit_transform(X)
X = X[:, 1:]
print("Transformed Value 0s and 1s Printed \n", X)

lc = LabelEncoder()
y = lc.fit_transform(y)
print("Last column encoded to ",y)


# splitting the Data in to Train and Test

from sklearn.model_selection import train_test_split

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.25,random_state=150)
print("Training X Value",X_train)
print("Training y Value",y_train)
print("Test X Value",X_test)
print("Test y Value",y_test)



# 1st Model LR
from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
Regressor.fit(X_train, X_test)
print("Intercept", Regressor.intercept_)
print("Slope", Regressor.coef_)

y_pred = Regressor.predict(y_train)
result_df = pd.DataFrame({'Actual Value on Y_test': y_test, 'Predict  Value on Y_pred': y_pred})
print("Data Frame Created for = ", result_df)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error", mse)
rmse = np.sqrt(mse)
print("Root Mean Square", rmse)
R_square = r2_score(y_test, y_pred)
print("R-Squared Value", R_square)
