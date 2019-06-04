# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:40:47 2019

@author: Hugo Borges
"""
#fonte do tutorial
#http://mundoia.com.br/tutorial/conheca-o-kaggle-e-participe-da-sua-primeira-competicao-de-machine-learning/


import pandas as pd
import matplotlib.pyplot as plt


 
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