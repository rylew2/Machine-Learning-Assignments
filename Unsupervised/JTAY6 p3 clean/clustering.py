#################################################################################
# CS 7641 - Assignment 3
# Ryan Lewis
# Code adapted from Jontay (https://github.com/JonathanTay/CS-7641-assignment-3)
#####################################################################################


#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}/'.format(sys.argv[1])

np.random.seed(0)
# digits = pd.read_hdf(out+'datasets.hdf','digits')
# digitsX = digits.drop('Class',1).copy().values
# digitsY = digits['Class'].copy().values
#
# madelon = pd.read_hdf(out+'datasets.hdf','madelon')
# madelonX = madelon.drop('Class',1).copy().values
# madelonY = madelon['Class'].copy().values

if sys.argv[1] == "BASE":
    column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]

    abalone = pd.read_csv("abalone.data", names=column_names)
    wine = pd.read_csv('winequality-white.csv', sep=';')

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

    wineX = wine.drop('quality', 1).copy().values
    wineY1 = wine['quality'].copy().values
    wineY = wineY1 <= 5

    abaloneX = StandardScaler().fit_transform(abaloneX)
    wineX = StandardScaler().fit_transform(wineX)

else:
    # abalone = pd.read_csv('./{}/abalone.csv'.format(sys.argv[1]))
    abalone = pd.read_csv('./{}/abalone.csv'.format(sys.argv[1]))
    wine = pd.read_csv('./{}/wine.csv'.format(sys.argv[1]))

    abalone = abalone.loc[:, ~abalone.columns.str.contains('^Unnamed')]
    wine = wine.loc[:, ~wine.columns.str.contains('^Unnamed')]

    abaloneX = abalone
    abaloneX = abaloneX.drop('Class', 1)
    abaloneX = abaloneX.astype(np.float64)
    abaloneY = abalone['Class']

    wineX = wine.drop('Class', 1).copy().values
    wineY = wine['Class'].copy().values


clusters =  [2,5,10,15,20,25,30,35,40]
# clusters = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]
# clusters =  [10, 50, 100, 500, 1000, 2000, 2780 ]

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(abaloneX)
    gmm.fit(abaloneX)
    SSE[k]['abalone'] = km.score(abaloneX)
    ll[k]['abalone'] = gmm.score(abaloneX)


    acc[k]['abalone']['Kmeans'] = cluster_acc(abaloneY,km.predict(abaloneX))
    acc[k]['abalone']['GMM'] = cluster_acc(abaloneY,gmm.predict(abaloneX))
    adjMI[k]['abalone']['Kmeans'] = ami(abaloneY,km.predict(abaloneX))
    adjMI[k]['abalone']['GMM'] = ami(abaloneY,gmm.predict(abaloneX))

    km.fit(wineX)
    gmm.fit(wineX)
    SSE[k]['wine'] = km.score(wineX)
    ll[k]['wine'] = gmm.score(wineX)
    acc[k]['wine']['Kmeans'] = cluster_acc(wineY,km.predict(wineX))
    acc[k]['wine']['GMM'] = cluster_acc(wineY,gmm.predict(wineX))
    adjMI[k]['wine']['Kmeans'] = ami(wineY,km.predict(wineX))
    adjMI[k]['wine']['GMM'] = ami(wineY,gmm.predict(wineX))
    print(k, clock()-st)
    if k==2:
        km.fit(abaloneX)
        np.savetxt(out+"abaloneClusterLabelsKMeans.csv", km.predict(abaloneX).astype(int), delimiter="," )
        gmm.fit(abaloneX)
        np.savetxt(out + "abaloneClusterLabelsGMM.csv", gmm.predict(abaloneX).astype(int), delimiter=",")



SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
acc.ix[:,:,'wine'].to_csv(out+'wine acc.csv')
acc.ix[:,:,'abalone'].to_csv(out+'abalone acc.csv')
adjMI.ix[:,:,'wine'].to_csv(out+'wine adjMI.csv')
adjMI.ix[:,:,'abalone'].to_csv(out+'abalone adjMI.csv')


#%% NN fit data (2,3)

# grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch, 'km__n_clusters':clusters}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(n_clusters=2, random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
# pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone cluster Kmeans.csv')


# grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
# pipe = Pipeline([('gmm',gmm),('NN',mlp)])
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(abaloneX,abaloneY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'abalone cluster GMM.csv')




# grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
# pipe = Pipeline([('km',km),('NN',mlp)])
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine cluster GMM.csv')



# %% For chart 4/5
# ncomponents = 3
#
# abaloneX2D = TSNE(n_components=ncomponents, verbose=10,random_state=5).fit_transform(abaloneX)
# wineX2D = TSNE(n_components=ncomponents, verbose=10,random_state=5).fit_transform(wineX)
#
# abalone2D = pd.DataFrame(np.hstack((abaloneX2D,np.atleast_2d(abaloneY).T)),columns=['x','y', 'z', 'target'])
# wine2D = pd.DataFrame(np.hstack((wineX2D,np.atleast_2d(wineY).T)),columns=['x','y', 'z', 'target'])
#
# abalone2D.to_csv(out+'abalone2D.csv')
# wine2D.to_csv(out+'wine2D.csv')


