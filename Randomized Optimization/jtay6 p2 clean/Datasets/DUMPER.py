# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np

import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

adult = pd.read_csv('winequality-white.csv', sep=';')
X = adult.drop('quality', 1).copy().values
adultY1 = adult['quality'].copy().values
adultY = adultY1 <= 5



from sklearn.preprocessing import scale
adultX = scale(X)


# madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)
madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)
pipe = Pipeline([('Scale',StandardScaler())])

trgX = pipe.fit_transform(madelon_trgX,madelon_trgY)
trgY = np.atleast_2d(madelon_trgY).T
tstX = pipe.transform(madelon_tstX)
tstY = np.atleast_2d(madelon_tstY).T


trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1,stratify=trgY)



tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))


print(tst.shape)
print(trg.shape)
print(val.shape)


tst.to_csv('m_test.csv',index=False,header=False)
trg.to_csv('m_trg.csv',index=False,header=False)
val.to_csv('m_val.csv',index=False,header=False)