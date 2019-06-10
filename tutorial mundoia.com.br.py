# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:40:47 2019

@author: Hugo Borges
"""
#fonte do tutorial
#http://mundoia.com.br/tutorial/conheca-o-kaggle-e-participe-da-sua-primeira-competicao-de-machine-learning/


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #import de regressão logistica
from sklearn.metrics import accuracy_score #avaliação do resultado
from sklearn.model_selection import cross_val_score #para validação cruzada

#importando dados
#fonte: https://www.kaggle.com/c/titanic/data
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

 
#verificar as dimensões dos dataframes
print("dimensões do dataset de treinameno: " + str(train.shape))
print("dimensões do dataset de teste: " + str(test.shape))


train.head()


#verificando a coluna sexo
sex_pivot = train.pivot_table(index="Sex",values="Survived") 
sex_pivot.plot.bar()
plt.show()


#verificando a coluna Pclass
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar(color='r') # r para indicar a cor vermelha(red)
plt.show()


#verificando a distribuição de idades no treino
train["Age"].describe()


#criando um histograma para visualizar como foi o grau de sobrevivência de acordo com as idades
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


#para facilitar o trabalhodo algoritmo, vamos criar ranges fixos de idades. 
# e ao mesmo tempo vamos tratar os missing values
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar(color='g')
plt.show()

#removendo a relação numerica presente na coluna P class
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df
 
for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)
    
    
#criação do modelo de machine learning com regressão logistica

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
'Age_categories_Missing','Age_categories_Infant', 'Age_categories_Child',
'Age_categories_Teenager', 'Age_categories_Young Adult', 'Age_categories_Adult',
'Age_categories_Senior']

#criando um objeto LogistcRegression
lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)

#avaliando o modelo
holdout = test

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
all_X, all_y, test_size=0.20,random_state=0)

lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)
print("acurácia com 20% do conjunto de teste: " + str(accuracy) + "\n")


#usando cross validation para um medida de erro mais precisa

scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print("acurácia das 10 subdivisões do conjunto de teste: " + str(scores) + "\n")
print("acurácia com 100% do conjunto de teste por validação cruzada com K-fold e 10 subdivisões: " + str(accuracy))


#fazendo previsões usando novos dados
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])