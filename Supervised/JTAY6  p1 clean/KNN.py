# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from matplotlib import pyplot as plt

dataset = 'abalone'
# dataset = 'wine'

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
# # fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
# axs[0].hist(adultY1, bins=n_bins)
# axs[1].hist(adultY, bins=n_bins)
# plt.xlabel('f', 0)
# plt.xlabel('axis label 1',1)
# plt.ylabel('axis label 1',1)
# plt.show()


from sklearn.preprocessing import scale
adultX = scale(X)

adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)

d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]

pipeA = Pipeline([('Scale',StandardScaler()), ('KNN',knnC())])

params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev', 'minkowski'],'KNN__n_neighbors':np.arange(1,50,2),'KNN__weights':['uniform','distance']}
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')

adult_final_params=adult_clf.best_params_

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')