# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:51:44 2018

@author: Natalie Menato
"""
import titanic
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


passengers = pd.read_csv('../data/train.csv')
passengers_test = pd.read_csv('../data/test.csv')
passengers = create_features(passengers)
passengers_test = create_features(passengers_test)

#Select desired features
features_to_use = ['Pclass', 'Sex', 'Age_binned', 'SibSp', 'Parch', 'Fare_binned', 'has_cabin', 'Embarked']
features_categorical = ['Sex', 'Pclass', 'Age_binned', 'Fare_binned', 'has_cabin', 'Embarked']

X = passengers[features_to_use]
X = pd.get_dummies(X, columns = features_categorical, drop_first = True)
y = passengers['Survived']

#split into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Random Forest
rf_clf = RandomForestClassifier().fit(X_train, y_train)
y_results = rf_clf.predict(X_test)
y_score = rf_clf.predict_proba(X_test)[:,1]

rf_clf.score(X_test, y_test) #.75336
roc_auc_score(y_test, y_score) #.81119

#Logistic Regression
lr_clf = LogisticRegression(solver = 'newton-cg').fit(X_train, y_train)
lr_clf.score(X_test, y_test) #.77578
roc_auc_score(y_test, lr_clf.decision_function(X_test)) #.87158


#SGDClassifier
sgd_clf = SGDClassifier(shuffle=True, loss = 'log').fit(X_train, y_train)
y_results = sgd_clf.predict(X_test)
y_score = sgd_clf.predict_proba(X_test)[:,1]

sgd_clf.score(X_test, y_test) #modified_huber .74439 log .70852
roc_auc_score(y_test, y_score) #modified_huber .74797 log .82314


#Support Vector Machine
svm_clf = svm.SVC(kernel='sigmoid').fit(X_train, y_train)
y_results = svm_clf.predict(X_test)

svm_clf.score(X_test, y_test) # linear .78475 rbf .78475
roc_auc_score(y_test, svm_clf.decision_function(X_test)) #linear .85864 rbf .81546 #sigmoid .82594

#Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier().fit(X_train, y_train)
y_results = gb_clf.predict(X_test)

gb_clf.score(X_test, y_test) # .77578
roc_auc_score(y_test, gb_clf.decision_function(X_test)) #linear .84893

X_final = passengers_test[features_to_use]
X_final = pd.get_dummies(X_final, columns = features_categorical, drop_first = True)
y_pred = lr_clf.predict(X_final)
results = pd.concat((passengers_test['PassengerId'], pd.Series(y_pred, name = 'Survived')), axis = 1)
results = results.set_index('PassengerId')
results.to_csv('../data/results.csv')





