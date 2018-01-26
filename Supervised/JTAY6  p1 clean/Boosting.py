# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""


import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# dataset = 'abalone'
dataset = 'wine'

if(dataset == 'wine'):
    adult = pd.read_csv('winequality-white.csv', sep = ';')
    X = adult.drop('quality', 1).copy().values
    adultY1 = adult['quality'].copy().values
    adultY = adultY1 <= 5
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

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     



adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)
# paramsA= {'Boost__n_estimators':[1,2,3,4,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

paramsA= {'Boost__n_estimators':[75, 100],'Boost__learning_rate':[1,.64,.32, .16, .08, .04, .02, .01, .001]}
paramsA= {'Boost__n_estimators':[1,2,3,4,5, 10,20,30,40,50, 60],'Boost__learning_rate':[1,.64,.32, .16, .08, .04, .02, .01, .001]}


# paramsA= {'Boost__n_estimators':[200, 400, 800, 1000],
#           'Boost__base_estimator__alpha':alphas}


         

adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)


pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',adult_booster)])

#
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,paramsA,'Boost','adult')        

#
#
#madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#

pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')
pipeA.set_params(**OF_params)

             
