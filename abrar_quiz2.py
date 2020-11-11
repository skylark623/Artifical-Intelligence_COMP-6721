# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:36:38 2020

@author: abrar
"""

import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
dataset = np.array([[1,1,0,1,0],
[1,1,1,0,1],
[0,0,1,0,0],
[0,1,0,1,0],
[1,0,1,1,1],
[0,1,1,1,0],])
X = dataset[:, 0:4]
y = dataset[:, 4]

dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X, y)

import graphviz
dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=['A last year', 'black hair', 'works hard', 'drinks'],
class_names=['A grade this year', 'Dont get A this year'],
filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("mytree")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
y_pred = dtc.predict([[0, 0, 0, 1]])
print( y_pred)