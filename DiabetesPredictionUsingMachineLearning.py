**Import libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""**Loading dataset**"""

df = pd.read_csv("diabetes.csv")

"""**Exploring data**"""

df.head(5)

df.tail(5)

df.sample(10)

df.shape

df.dtypes

df.info()

df.describe()

"""**Data Cleaning**

Drop the duplicate
"""

df.shape

df=df.drop_duplicates()

df.shape

"""**Check the null values**"""

df.isnull().sum()

df.columns

"""Check the number of zero values in the dataset"""

print('No. of zero values in Glucose',df[df['Glucose']==0].shape[0])

print('No. of zero values in BloodPressure',df[df['BloodPressure']==0].shape[0])

print('No. of zero values in SkinThickness',df[df['SkinThickness']==0].shape[0])

print('No. of zero values in Insulin',df[df['Insulin']==0].shape[0])

print('No. of zero values in BMI',df[df['BMI']==0].shape[0])

print('No. of zero values in Pregnancies',df[df['Pregnancies']==0].shape[0])

"""Replace number of zero with mean of the column"""

df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['Pregnancies']=df['Pregnancies'].replace(0,df['Pregnancies'].mean())

df.describe()

"""**Data Visualization**

Count plot
"""

# # DATA VISUALIZATION
# outcome count plot
f,ax=plt.subplots(1,2,figsize=(10,5))

df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%2.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')



ax[1].set_title('Outcome')

N,P = df['Outcome'].value_counts()
print('Negative (0): ',N)
print('Positive (1): ',P)
plt.grid()
plt. show()

"""Histograms"""

# Histogram of each feature
df.hist(bins=10, figsize=(10,10))
plt.show()

"""Scatter Plot"""

# Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize = (20, 20));

"""Pair Plot"""

#Pairplot
sns.pairplot(data = df, hue = 'Outcome')
plt.show()

"""**Analyzing relationships between variables**

CoRelation Ananlysis
"""

import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

"""**Split the data frame into X and Y**"""

target_name = 'Outcome'
Y = df[target_name]
X = df.drop(target_name,axis=1)

X.head()

Y.head()

"""**Applying features Scalling**"""

# StandardScaler can be applied
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
SSX = scaler.transform(X)

"""**Train test split**"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(SSX, Y, test_size=0.2, random_state=7)

X_train.shape,Y_train.shape

X_test.shape,Y_test.shape

"""**Build Classification Algorithm**

**1. Logistic Regresssion**
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train,Y_train)

"""**2. KNN**"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

"""**3. Naive-Bayes Classifier**"""

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)

"""**4. Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)

"""**Making Prediction**

**1. Logistic Regression**
"""

X_test.shape

lr_pred=lr.predict(X_test)

lr_pred.shape

"""**2. KNN**"""

knn_pred=knn.predict(X_test)

"""**3. Naive Bayes**"""

nb_pred=nb.predict(X_test)

"""**4. Decision Tree**"""

dt_pred=dt.predict(X_test)

"""**Model Evaluation**

**Train Score & Test Score**
"""

# Logistic Regression
from sklearn.metrics import accuracy_score
print("Train Accuracy of Logistic Regression",lr.score(X_train,Y_train)*100)
print("Accuracy Test score of Logistic Regression",lr.score(X_test,Y_test)*100)
print("Accuracy score of Logistic Regression",accuracy_score(Y_test,lr_pred)*100)

# KNN
from sklearn.metrics import accuracy_score
print("Train Accuracy of KNN",knn.score(X_train,Y_train)*100)
print("Accuracy Test score of KNN",knn.score(X_test,Y_test)*100)
print("Accuracy score of KNN",accuracy_score(Y_test,knn_pred)*100)

# Naive-Bayes
from sklearn.metrics import accuracy_score
print("Train Accuracy of Naive-Bayes",nb.score(X_train,Y_train)*100)
print("Accuracy Test score of Naive-Bayes",nb.score(X_test,Y_test)*100)
print("Accuracy score of Naive-Bayes",accuracy_score(Y_test,nb_pred)*100)

# Decision Tree
from sklearn.metrics import accuracy_score
print("Train Accuracy of Decision Tree",dt.score(X_train,Y_train)*100)
print("Accuracy Test score of Decision Tree",dt.score(X_test,Y_test)*100)
print("Accuracy score of Decision Tree",accuracy_score(Y_test,dt_pred)*100)

algorithm_names = ['Logistic Regression', 'KNN', ' Naive-Bayes', 'Decision Tree']
accuracy_scores = [accuracy_score(Y_test,lr_pred)*100, accuracy_score(Y_test,nb_pred)*100, accuracy_score(Y_test,knn_pred)*100, accuracy_score(Y_test,dt_pred)*100]
plt.bar(algorithm_names, accuracy_scores, width=0.3)
plt.xlabel('ML Algorithm')
plt.ylabel('Accuracy')
plt.title('ML Algorithm Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylim([0, 100])
plt.tight_layout()
plt.show()

"""**Obtain Prediction for Model**"""

predictedData = dt.predict(X_test)
predictedData

"""**Visualization of Decision Tree**"""

import graphviz
import matplotlib.pyplot as plt
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1000)
tree.plot_tree(dt);

fn=['	Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
cn=['diabetic','non diabetic']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1000)

tree.plot_tree(dt,
                feature_names = fn,
                class_names=cn,
                filled = True, rounded=True);

"""**Confusion Matrix of Logistic Regression**"""

from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(Y_test,lr_pred)
cm

sns.heatmap(confusion_matrix(Y_test,lr_pred),annot=True,fmt='d')

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
cm=confusion_matrix(Y_test,lr_pred)

print('TN - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print('Classification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))

print("Classification report of  Logistic Regression: \n",   classification_report(Y_test,lr_pred,digits=4))

TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

TN, FP, FN, TP

import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title("Confusion matrix of Logistic Regression")
plt.ylabel("Actual (True) values")
plt.xlabel("Predicted values")
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'],['FN','TP']]
for i in range(2):
  for j in range(2):
    plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

"""**Precision (PPV - Positive Predictive Value)**"""

TP, FP

Precision = TP/(TP+FP)
Precision

33/(33+11)

print('Classification Report of  Logistic Regression : \n',classification_report(Y_test,lr_pred,digits=4))

"""**Recall (TPR - True Postive Rate)**"""

recall_score = TP / float(TP + FN)*100
print('recall_score',recall_score)

TP, FN

33/(33+24)

from sklearn.metrics import recall_score
print("Recall or Sensitivity_score : ", recall_score(Y_test,lr_pred)*100)

print("Classification Report of  Logistic Regression: \n",classification_report(Y_test,lr_pred, digits=4))

"""**False Positive Rate (FPR)**"""

FP,TN

11/11+86

"""**Specificity**

**F1-Score**
"""

from sklearn.metrics import f1_score
print('f1_score of macro: ',f1_score(Y_test, lr_pred)*100)

"""**Area Under Curve of Logistic Regression**"""

from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,lr_pred)*100,2)
print('roc_auc_score of Logistic Regression:',auc)

fpr, tpr, thresholds = roc_curve(Y_test, lr_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Logistic Regression')
plt.legend()
plt.grid()
plt.show()

"""**Confusion Matrix of KNN**"""

from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(Y_test,knn_pred)
cm

sns.heatmap(confusion_matrix(Y_test,knn_pred),annot=True,fmt='d')

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
cm=confusion_matrix(Y_test,knn_pred)

print('TN - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print('Classification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))

print("Classification report of KNN: \n",   classification_report(Y_test,knn_pred,digits=4))

TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

TN, FP, FN, TP

import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title("Confusion matrix of KNN")
plt.ylabel("Actual (True) values")
plt.xlabel("Predicted values")
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'],['FN','TP']]
for i in range(2):
  for j in range(2):
    plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

"""**Precision (PPV - Positive Predictive Value)**"""

TP, FP

Precision = TP/(TP+FP)
Precision

36/(36+16)

print('Classification Report of KNN: \n',classification_report(Y_test,knn_pred,digits=4))

"""**Recall (TPR - True Postive Rate)**"""

recall_score = TP / float(TP + FN)*100
print('recall_score',recall_score)

TP, FN

36/(36+21)

from sklearn.metrics import recall_score
print("Recall or Sensitivity_score : ", recall_score(Y_test,knn_pred)*100)

print("Classification Report of KNN: \n",classification_report(Y_test,knn_pred, digits=4))

"""**False Positive Rate (FPR)**"""

FP,TN

16/16+81

"""**Specificity**

**F1-Score**
"""

from sklearn.metrics import f1_score
print('f1_score of macro: ',f1_score(Y_test, knn_pred)*100)

"""**Area Under Curve of KNN**"""

from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,knn_pred)*100,2)
print('roc_auc_score of KNN:',auc)

fpr, tpr, thresholds = roc_curve(Y_test, knn_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of KNN')
plt.legend()
plt.grid()
plt.show()

"""**Confusion Matrix of Naive-Bayes**"""

from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(Y_test,nb_pred)
cm

sns.heatmap(confusion_matrix(Y_test,nb_pred),annot=True,fmt='d')

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
cm=confusion_matrix(Y_test,nb_pred)

print('TN - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print('Classification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))

print("Classification report of  Naive-Bayes: \n",   classification_report(Y_test,nb_pred,digits=4))

TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

TN, FP, FN, TP

import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title("Confusion matrix of  Naive-Bayes ")
plt.ylabel("Actual (True) values")
plt.xlabel("Predicted values")
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'],['FN','TP']]
for i in range(2):
  for j in range(2):
    plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

"""**Precision (PPV - Positive Predictive Value)**"""

TP, FP

Precision = TP/(TP+FP)
Precision

37/(37+18)

print('Classification Report of  Naive-Bayes: \n',classification_report(Y_test,nb_pred,digits=4))

"""**Recall (TPR - True Postive Rate)**"""

recall_score = TP / float(TP + FN)*100
print('recall_score',recall_score)

TP, FN

37/(37+20)

from sklearn.metrics import recall_score
print("Recall or Sensitivity_score : ", recall_score(Y_test,nb_pred)*100)

print("Classification Report of  Naive-Bayes: \n",classification_report(Y_test,nb_pred, digits=4))

"""**False Positive Rate (FPR)**"""

FP,TN

18/18+79

"""**Specificity**

**F1-Score**
"""

from sklearn.metrics import f1_score
print('f1_score of macro: ',f1_score(Y_test, nb_pred)*100)

"""**Area Under Curve of Naive-Bayes**"""

from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,nb_pred)*100,2)
print('roc_auc_score of  Naive-Bayes:',auc)

fpr, tpr, thresholds = roc_curve(Y_test, nb_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of  Naive-Bayes')
plt.legend()
plt.grid()
plt.show()

"""**Confusion Matrix of Decision Tree**"""

from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(Y_test,dt_pred)
cm

sns.heatmap(confusion_matrix(Y_test,dt_pred),annot=True,fmt='d')

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
cm=confusion_matrix(Y_test,dt_pred)

print('TN - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print('Classification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))

print("Classification report of Decision Tree: \n",   classification_report(Y_test,dt_pred,digits=4))

TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

TN, FP, FN, TP

import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title("Confusion matrix of Decision Tree")
plt.ylabel("Actual (True) values")
plt.xlabel("Predicted values")
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'],['FN','TP']]
for i in range(2):
  for j in range(2):
    plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

"""**Precision (PPV - Positive Predictive Value)**"""

TP, FP

Precision = TP/(TP+FP)
Precision

43/(43+17)

print('Classification Report of Decision Tree: \n',classification_report(Y_test,dt_pred,digits=4))

"""**Recall (TPR - True Postive Rate)**


"""

recall_score = TP / float(TP + FN)*100
print('recall_score',recall_score)

TP, FN

43/(43+14)

from sklearn.metrics import recall_score
print("Recall or Sensitivity_score : ", recall_score(Y_test,dt_pred)*100)

print("Classification Report of Decision Tree: \n",classification_report(Y_test,dt_pred, digits=4))

"""**False Positive Rate (FPR)**"""

FPR = FP / float(FP+TN)*100
print('False Positive Rate :{0:0.4f}'.format(FPR))

FP,TN

16/(16+81)

"""**Specificity**"""

specificity = TN/ (TN+FP)*100
print('Specificity :{0:0.4f}'.format(specificity))
a = 100 - specificity
print('FPR :',a)

"""**F1-Score**"""

from sklearn.metrics import f1_score
print('f1_score of macro: ',f1_score(Y_test, dt_pred)*100)

"""**Area Under Curve of Decision Tree**"""

from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,dt_pred)*100,2)
print('roc_auc_score of Decision Tree:',auc)

fpr, tpr, thresholds = roc_curve(Y_test, dt_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Decision Tree')
plt.legend()
plt.grid()
plt.show()

"""**Making Prediction**"""

# input_data = (6,148,72,35,0,33.6,0.627,50)
# input_data = (1,85,66,29,0,26.6,0.351,31)
# input_data = (2,102,86,36,120,45.5,0.127,23) #line310
input_data = (2,146,70,38,360,28,0.337,29) #line 298

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = dt.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')