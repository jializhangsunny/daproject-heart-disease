#import external libaries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.


#import data and data cleaning
datafile=pd.read_csv('heart.csv')
datafile.head()
datafile.duplicated()
print(datafile.head())
print(datafile.describe())
print(datafile.duplicated())

#explore the correlation
plt.figure(figsize=(20,10))
sns.heatmap(datafile.corr(), annot=True, cmap="YlGnBu", linewidths=1)
plt.show()

#explore different ages and heart disease number
pd.crosstab(datafile.age, datafile.target).plot(kind="bar", figsize=(25, 15), color=['turquoise', 'cornflowerblue'])
plt.xlabel('Age',fontsize=20, fontweight='bold')
plt.ylabel('Population contribution',fontsize=20, fontweight='bold')
plt.legend(["No Disease", "Have Disease"])
plt.show()

#explore different gender and heart disease number
pd.crosstab(datafile.sex, datafile.target).plot(kind="bar", figsize=(10, 5), color=['turquoise', 'cornflowerblue'])
plt.xlabel('Sex (0 = Female, 1 = Male)',fontsize=20, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Have Disease"])
plt.ylabel('Population contribution',fontsize=20, fontweight='bold')
plt.show()

#explore different gender number in different age who have heart disease
x=datafile.age[datafile.target == 1]
pd.crosstab(x, datafile.sex, ).plot(kind="bar", figsize=(25, 15), color=['pink', 'cornflowerblue'])
plt.xlabel('Age',fontsize=16, fontweight='bold')
plt.legend(['Female', 'Male'],fontsize=16)
plt.ylabel('Population contribution',fontsize=16, fontweight='bold')
plt.show()

#explore the the person's resting blood pressure based on age
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.scatter(x=datafile.age[datafile.target==1],y=datafile.trestbps[datafile.target==1],c='turquoise')
plt.scatter(x=datafile.age[datafile.target==0],y=datafile.trestbps[datafile.target==0],c='cornflowerblue')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.legend(['Disease','No Disease'])

#explore the the person's cholesterol measurement based on age
plt.subplot(1,2,2)
plt.scatter(x=datafile.age[datafile.target==1],y=datafile.chol[datafile.target==1],c='turquoise')
plt.scatter(x=datafile.age[datafile.target==0],y=datafile.chol[datafile.target==0],c= 'cornflowerblue')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend(['Disease','No Disease'])
plt.show()

#Creating Dummy Variables
chest_pain=pd.get_dummies(datafile['cp'], prefix='cp', drop_first=True)
datafile=pd.concat([datafile, chest_pain], axis=1)
datafile.drop(['cp'], axis=1, inplace=True)
sp=pd.get_dummies(datafile['slope'], prefix='slope')
th=pd.get_dummies(datafile['thal'], prefix='thal')
rest_ecg=pd.get_dummies(datafile['restecg'], prefix='restecg')
frames=[datafile, sp, th, rest_ecg]
datafile=pd.concat(frames, axis=1)
datafile.drop(['slope', 'thal', 'restecg'], axis=1, inplace=True)
datafile.head()
print(datafile.head())

#split the features and tartget values and preparing for leaning
X = datafile.drop(['target'], axis = 1)
y = datafile.target.values
print(X)
print(y)

#import libraries
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#neural network model
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense

# Adding the input layer and the first hidden layer
model.add(Dense(output_dim = 11, init ='uniform', activation ='relu', input_dim = 22))

# Adding the third hidden layer
model.add(Dense(output_dim = 11, init ='uniform', activation ='relu'))

# Adding the output layer
model.add(Dense(output_dim = 1, init ='uniform', activation ='sigmoid'))

# Compiling the ANN
model.compile(optimizer ='adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Choose the number of epochs
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = model.predict(X_test)
# print(y_pred)
# print(y_pred.round())

import seaborn as sns
from sklearn.metrics import confusion_matrix
nncm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(nncm, annot=True, cmap="YlGnBu", fmt="d", cbar=False)

#accuracy score
from sklearn.metrics import accuracy_score
nn_ac=accuracy_score(y_test, y_pred.round())
print('NeuralNetwork_accuracy:', nn_ac)
plt.show()

#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
rf_c=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_c.fit(X_train, y_train)
rf_pred=rf_c.predict(X_test)
rf_cm=confusion_matrix(y_test, rf_pred)
rdf_ac=accuracy_score(rf_pred, y_test)
sns.heatmap(rf_cm, annot=True, cmap="YlGnBu", fmt="d", cbar=False)
print('RandomForest Accuracy:',rdf_ac)
plt.show()

# svm
from sklearn.svm import SVC
svm = SVC(C=1, kernel='rbf', random_state=1)
svm.fit(X_train, y_train)
svm_pred=svm.predict(X_test)
accuracies = {}
svm_cm = confusion_matrix(y_test,svm_pred)
sns.heatmap(svm_cm,annot=True,cmap="YlGnBu",fmt="d",cbar=False)
svm_ac = svm.score(X_test, y_test)
accuracies['SVM'] = svm_ac
print('SVM_accuracy:', svm_ac)
plt.show()


# model accuracy
model_accuracy = pd.Series(data=[rdf_ac, nn_ac, svm_ac], index=['RandomForest', 'Neural Network', 'SVM'])
# fig= plt.figure(figsize=(10,5))
model_accuracy.sort_values().plot(kind='bar',figsize=(10,5),color='cornflowerblue')
plt.show()