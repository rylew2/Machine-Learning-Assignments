# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:55:32 2017

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
import numpy as np
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    return

# Load Data
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



from sklearn.preprocessing import scale
adultX = scale(X)
print(np.isfinite(adultX).all())
print(np.isfinite(adultY).all())

adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)

# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
depths = np.arange(1,10)
minSampleLeafs =  np.arange(1, 200, 50)


pipeA = Pipeline([('Scale',StandardScaler()),
                 ('DT',DecisionTreeClassifier(random_state=0) )])
params = {'DT__min_samples_leaf':minSampleLeafs, 'DT__max_depth':depths, 'DT__criterion':['gini','entropy'],'DT__class_weight':['balanced'], 'DT__splitter':['best', 'random']}

# if dataset!='aa': #use pruning tree and alpha
#     pipeA = Pipeline([('Scale',StandardScaler()),
#                      ('DT',dtclf_pruned(random_state=55))])
#     params = {'DT__min_samples_leaf':minSampleLeafs,'DT__max_depth':depths, 'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced'], 'DT__splitter':['best', 'random']}
#

adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params,'DT','adult')

adult_final_params = adult_clf.best_params_

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'DT','adult')

if dataset!='abalone':
    # DTpruningVSnodes(pipeA,alphas,adult_trgX,adult_trgY,'adult')
    zz=2