
import pandas as pd
link = "C:\\Users\\USER\\OneDrive\\Top_Mentor\\Classes\\DataSets\\MachineLearning-master\\5_Ads_Success.csv"
data = pd.read_csv(link)
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values
X_plt = data.iloc[:,2:-1].values
#handling categorical data - since gender has 2 values, encoding is enough
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
X[:,0] = lc.fit_transform(X[:,0])
print(X)

# breaking this data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
X_train_plt, X_test_plt, y_train_plt, y_test_plt = train_test_split(X_plt,y,test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)
X_train_plt = sc_x.fit_transform(X_train_plt)
## Run our classification model

# Model 2: K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
classifier_lr = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
classifier_lr.fit(X_train,y_train)


y_pred = classifier_lr.predict(X_test)
outcome_df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(outcome_df)

# visualization
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
classifier_lr_plt =  KNeighborsClassifier(n_neighbors=3)
classifier_lr_plt.fit(X_train_plt,y_train_plt)
# X data has 0 - gender, 1 - Age, 2 - salary
X_set, y_set = X_train, y_train
import numpy as np
X1, X2 = np.meshgrid(np.arange(start=X_set[:,1].min()-1,
                               stop=X_set[:,1].max()+1,
                               step = 0.01),
                     np.arange(start=X_set[:,2].min()-1,
                               stop=X_set[:,2].max()+1,
                               step = 0.01))
plt.contourf(X1,X2, classifier_lr_plt.predict(np.array([X1.ravel(),
                                        X2.ravel()]).T).reshape(X1.shape),
                                        cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("K Nearest Neighbor")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix = \n",cm)
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
print("Accuracy = ",(TP+TN)/(TP+TN+FP+FN))
#calculate other metrics using above metric

print("Accuracy =",metrics.accuracy_score(y_test, y_pred))
print("Accuracy =",metrics.f1_score(y_test, y_pred))
print("Accuracy =",metrics.recall_score(y_test, y_pred))

## Plotting ROC curve
fpr, tpr, thresolds = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.show()

# calculate AUC - area under ROC curve
print("AUC = ",metrics.roc_auc_score(y_test, y_pred))

# calculating distances
distances = []

for i in range(1,16,2):
    classifier_lr = KNeighborsClassifier(n_neighbors=i, metric='minkowski',p=2)
    classifier_lr.fit(X_train, y_train)
    labels = classifier_lr.predict(X_train)
    distances.append(np.sum((labels-y_train)**2))
# plotting of distances
plt.plot(range(1,16,2), distances)
plt.show()
