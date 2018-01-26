
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.svm

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred
    



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

    adultY1 = adult['rings']
    adultY = adultY1 <=9


from sklearn.preprocessing import scale
adultX = scale(X)


adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     

N_adult = adult_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#RBF SVM
gamma_fracsA = np.arange(0.2,2.1,0.2)

#
# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('SVM',primalSVM_RBF())])


#dict_keys(['C', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])


pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',sklearn.svm.SVC())])

# params_adult = {'SVM__kernel':['rbf'],'SVM__gamma':[ 0.2,  0.4,  0.6,  0.8,  1 ], 'SVM__C':[  1.00000000e-02,   1.00000000e-01,   1.00000000e+00, 1.00000000e+01, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]}
params_adult = {'SVM__kernel':['linear','rbf', 'poly'],'SVM__gamma':[ 0.2,  0.4,  0.6,  0.8,  1], 'SVM__C':[  1.00000000e-02,   1.00000000e-01,   1.00000000e+00]}

# params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}

adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_RBF','adult')

adult_final_params =adult_clf.best_params_
# adult_OF_params = adult_final_params.copy()
# adult_OF_params['SVM__alpha'] = 1e-16

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'SVM_RBF','adult')


pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__max_iter':np.arange(1,75,3)},'SVM_RBF','adult')

# pipeA.set_params(**adult_OF_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult')


#http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
#http://scikit-learn.org/stable/modules/svm.html#svm-kernels
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html