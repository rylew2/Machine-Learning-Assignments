#################################################################################
# Ryan Lewis
# Code adapted from Jontay
#####################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'

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

clusters =  [2,5,10,15,20,25,30,35,40]

abalonedims = [1,2,3,4,5,6,7,8,9,10]
winedims = [1,2,3,4,5,6,7,8,9,10, 11]

#%% data for 1
pca = PCA(random_state=0)
pca.fit(abaloneX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,11))
tmp.to_csv(out+'abalone scree.csv')


pca = PCA(random_state=0)
pca.fit(wineX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,12))
tmp.to_csv(out+'wine scree.csv')


#%% Data for 2

grid ={'pca__n_components':abalonedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=0)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=0)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone dim red.csv')


grid ={'pca__n_components':winedims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=0)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=0)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up

dim = 7
pca = PCA(n_components=dim,random_state=0)
abaloneX2 = pca.fit_transform(abaloneX)
abalone2 = pd.DataFrame(np.hstack((abaloneX2,np.atleast_2d(abaloneY).T)))
cols = list(range(abalone2.shape[1]))
cols[-1] = 'Class'
abalone2.columns = cols
abalone2.to_csv(out+'abalone.csv')

dim = 8
pca = PCA(n_components=dim,random_state=0)
wineX2 = pca.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine.csv')