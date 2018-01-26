# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)


column_names = ["sex", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]
abalone = pd.read_csv("abalone.data", names=column_names)
abalone.fillna(abalone.mean())

for label in "MFI":
    abalone[label] = abalone["sex"] == label
del abalone["sex"]
abaloneX = abalone
abaloneX = abaloneX.drop('rings', 1)

abaloneX['M'] = abaloneX['M'].astype(int)
abaloneX['F'] = abaloneX['F'].astype(int)
abaloneX['I'] = abaloneX['I'].astype(int)
abaloneX = abaloneX.astype(np.float64)

abaloneY1 = abalone['rings']
abaloneY = abaloneY1 <= 9


wine = pd.read_csv('winequality-white.csv', sep=';')
wineX = wine.drop('quality', 1).copy().values
wineY1 = wine['quality'].copy().values
wineY = wineY1 <= 5


abaloneX = StandardScaler().fit_transform(abaloneX)
wineX = StandardScaler().fit_transform(wineX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine NN bmk.csv')
# raise