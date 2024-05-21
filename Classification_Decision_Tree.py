import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,accuracy_score

path = "C:\\Users\\USER\\OneDrive\\Top_Mentor\\Classes\\DataSets\\MachineLearning-master\\5_Ads_Success.csv"
dataset = pd.read_csv(path)
print("Data Fetched from Local", dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lc = LabelEncoder()
X[:, 1] = lc.fit_transform(X[:, 1])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]
print(X)

# breaking this data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=100)
print("X_train", X_train)
print("y_train", y_train)
print("X_test", X_test)
print("y_test", y_test)


# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

Regressor = DecisionTreeClassifier(criterion="entropy")
Regressor = Regressor.fit(X_train, y_train)

y_pred = Regressor.predict(X_test)
outcome_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(outcome_df)



### visualize the tree format
from sklearn import tree

fig = plt.figure(figsize=(20,30))
_ = tree.plot_tree(Regressor)
plt.show()

### visualize the tree data
tree_form = tree.export_text(Regressor)
print(tree_form)



# Model Evaluation

from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix = \n", cm)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]




# Specificity
# =============
specificity = TN / (TN + FP)
print("Specificity Value", specificity)

## Plotting ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.show()

print("AUC = ", metrics.roc_auc_score(y_test, y_pred))

# calculate AUC - area under ROC curve
# Prevalence: Actual Yes / Total

# Prevalence = Actual Positive  / Total Observations

# Formula :
# prevalence = FN + TP  / TN + FP + FN + TP

# prevalence =  (0 + 2) / (7 + 1 + 0 + 2)

prevalence = (FN + TP) / (TN + FP + FN + TP)

print("Prevalence", prevalence)

# Miss-Classification Rate:

# Miss_classification = False Predictions / Total Observations

Miss_classification_rate = (FP + FN) / (TN + FP + FN + TP)

print("Miss_classification_rate", Miss_classification_rate)


#Analyze
import numpy as np
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
#RMSE & R2
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE = ",rmse)
r2 = metrics.r2_score(y_test, y_pred)
print("R Square = ",r2)

accuracy_score=metrics.accuracy_score(y_test,y_pred)
print("Accuracy Value",accuracy_score)
F1_score = metrics.f1_score(y_test,y_pred)
print("F1 Score Value",F1_score)
Recall = metrics.recall_score(y_test,y_pred)
print("Recall Value",Recall)
