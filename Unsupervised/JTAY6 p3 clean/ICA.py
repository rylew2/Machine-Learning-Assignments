#################################################################################
# Ryan Lewis
# Code adapted from Jontay
#####################################################################################


#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './ICA/'

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
wineX= StandardScaler().fit_transform(wineX)



dims =  [2,5,10,15,20,25,30,35,40]
# dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
# dims = [1,2,3,4,5,6,7,8,9,10]

abalonedims = [1,2,3,4,5,6,7,8,9,10]
winedims = [1,2,3,4,5,6,7,8,9,10, 11]

max_iters=1000
tolerance = .5
#raise
#%% data for 1

ica = FastICA(random_state=5)
kurt = {}
for dim in abalonedims:
    ica.set_params(n_components=dim, max_iter=max_iters, tol=tolerance)
    tmp = ica.fit_transform(abaloneX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'abalone scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in winedims:
    ica.set_params(n_components=dim, max_iter=max_iters, tol=tolerance)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'wine scree.csv')
# raise

#%% Data for 2

grid ={'ica__n_components':abalonedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5, max_iter=max_iters, tol=tolerance)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'ica__n_components':winedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5, max_iter=max_iters, tol=tolerance)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')
# raise

#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up


dim = 4
ica = FastICA(n_components=dim,random_state=10, max_iter=max_iters, tol=tolerance)

abaloneX2 = ica.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv(out+'abalone.csv')


dim = 11
ica = FastICA(n_components=dim,random_state=10, max_iter=max_iters, tol=tolerance)
wineX2 = ica.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine.csv')