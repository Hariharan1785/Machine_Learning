import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error

#link = "C:\\Users\\USER\\OneDrive\\Top_Mentor\\Classes\\DataSets\\MachineLearning-master\\3_Startups.csv"
link = "https://raw.githubusercontent.com/Hariharan1785/Machine_Learning/main/3_Startups.csv"
mul_dataset = pd.read_csv(link)
print("", mul_dataset)
print(mul_dataset.shape)
plt.scatter(mul_dataset["Marketing Spend"], mul_dataset["Profit"])
plt.show()

X = mul_dataset.iloc[:, :-1].values
y = mul_dataset.iloc[:, -1].values
print("Expect Last Column printed \n", X)
print("Last Column Printed \n", y)
# handling missing values
import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 0:2])
X[:, 0:2] = imputer.transform(X[:, 0:2])
print(X)

# handling categorical data coonverting  text to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lc = LabelEncoder()
X[:, 3] = lc.fit_transform(X[:, 3])
print("################# 3rd Column Transfer using Label Encoder #################### \n", lc)
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
print("################Column Transfer Converted 0s and 1s ################################ \n", ct)
X = ct.fit_transform(X)
X = X[:, 1:]
print("Transformed Value Printed \n", X)

from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
y_train = sc_x.fit_transform(y_train)
print("After scaling X_train data: \n", X_train)
print("After scaling X_test data: \n", y_train)

# OLS method - regression using OLS
# X_train, X_test, y_train, y_test
from statsmodels.api import OLS
import statsmodels.api as sm

X = np.array(X, dtype=float)
X = sm.add_constant(X)
print("============================\nX:\n", X)
summary = OLS(y, X).fit().summary()
print("Summary - All X are present\n", summary)

# performing Backward elimination method to know which columns are important
# remove X2 as it has a p value of 0.990
X_1 = X[:, [0, 1, 3, 4, 5]]
print("============== X2 is removed ==============")
summary = OLS(y, X_1).fit().summary()
print("Summary - All X are present\n", summary)

# performing Backward elimination method to know which columns are important
# remove X2 as it has a p value of 0.990
X_1 = X[:, [0, 3, 4, 5]]
print("============== X1 & X2 is removed ==============")
summary = OLS(y, X_1).fit().summary()
print("Summary - All X are present\n", summary)

X_1 = X[:, [0, 3, 5]]
print("============== X1 & X2 & X4 removed ==============")
summary = OLS(y, X_1).fit().summary()
print("Summary - All X are present\n", summary)

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


