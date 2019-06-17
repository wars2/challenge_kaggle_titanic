# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:51:21 2019

@author: Microtc
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#retirando os dados irrelevantes
train.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)
test.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)

#fazendo uso dos dummies de novo
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

new_data_train.isnull().sum().sort_values(ascending = False).head(10)

#tratando valores nulos encontrados
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace = True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace = True)

new_data_test.isnull().sum().sort_values(ascending = False).head(10)


new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace = True)

#separado as features para a criação do modelo
X = new_data_train.drop("Survived", axis = 1) #tirando apenas a coluna target 
y = new_data_train["Survived"] # colocando somente a coluna target

tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)
tree.fit(X,y)

#avaliando o modelo
tree.score(X,y)