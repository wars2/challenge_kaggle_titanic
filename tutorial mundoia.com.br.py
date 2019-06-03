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

#verificando a coluna sexo
sex_pivot = train.pivot_table(index="Sex",values="Survived") 
sex_pivot.plot.bar()
plt.show()

#verificando a coluna Pclass
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar(color='r') # r para indicar a cor vermelha(red)
plt.show()