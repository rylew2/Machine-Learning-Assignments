#################################################################################
# Ryan Lewis
# Code adapted from Jontay
#####################################################################################


#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './RP/'
cmap = cm.get_cmap('Spectral')

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

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [1,2,3,4,5,6,7,8,9,10]
abalonedims = [1,2,3,4,5,6,7,8,9,10]
winedims = [1,2,3,4,5,6,7,8,9,10, 11]


#raise
#%% data for 1

tmp = defaultdict(dict)
for i,dim in product(range(10),abalonedims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(abaloneX), abaloneX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'abalone scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),winedims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(wineX), wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),abalonedims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(abaloneX)
    tmp[dim][i] = reconstructionError(rp, abaloneX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'abalone scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),winedims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)
    tmp[dim][i] = reconstructionError(rp, wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree2.csv')

#%% Data for 2

grid ={'rp__n_components':abalonedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'rp__n_components':winedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')
# raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 7
rp = SparseRandomProjection(n_components=dim,random_state=5)

abaloneX2 = rp.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv(out+'abalone.csv')

dim = 8
rp = SparseRandomProjection(n_components=dim,random_state=5)
wineX2 = rp.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine.csv')