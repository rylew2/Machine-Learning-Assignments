# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# dataset = 'abalone'
dataset = 'wine'

if(dataset == 'wine'):
    adult = pd.read_csv('winequality-white.csv', sep = ';')
    X = adult.drop('quality', 1).copy().values
    adultY1 = adult['quality'].copy().values
    adultY = adultY1 <= 5
    zz=2
elif(dataset == 'abalone'):
    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]
    adult = pd.read_csv("abalone.data", names=column_names)
    adult.fillna(adult.mean())

    for label in "MFI":
        adult[label] = adult["sex"] == label
    del adult["sex"]
    X = adult
    X = X.drop('rings', 1)

    X['M'] = X['M'].astype(int)
    X['F'] = X['F'].astype(int)
    X['I'] = X['I'].astype(int)
    X = X.astype(np.float64)
    # X += np.random.randn(X.shape[0], X.shape[1]) * 0.00001
    # X.replace([np.inf, -np.inf], np.nan)

    adultY1 = adult['rings']

    # adultY1[adultY1 < 6] = 0 #3 class classification
    # adultY1[adultY1 >12] =2
    # adultY1[adultY1>=6] = 1
    # adultY = adultY1
    adultY = adultY1 <=9



# n_bins = 10
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(adultY1, bins=n_bins)
# axs[1].hist(adultY, bins=n_bins)
# plt.show()


from sklearn.preprocessing import scale
adultX = scale(X)

adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)


# z=  MLPClassifier(max_iter=2000,early_stopping=True,random_state=55)
#
# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
#

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=10000,early_stopping=False,random_state=55))])



d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
# alphas = [0.1]
# alphas.extend([ 1e-8,  1e-10, 1e-12])

params_adult = {'MLP__activation':['relu','logistic','identity','tanh'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
# params_adult = {'MLP__activation':['tanh'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}

#

adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'ANN','adult')        


#madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
#adult_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}

adult_final_params =adult_clf.best_params_
adult_OF_params =adult_final_params.copy()
adult_OF_params['MLP__alpha'] = 0


#raise#

pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')


pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','adult')                


pipeA.set_params(**adult_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','adult')

