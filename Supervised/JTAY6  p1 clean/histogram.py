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
import matplotlib.mlab as mlab

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

if dataset=='wine':
    n_bins = 9
    plt.hist(adultY1, bins=n_bins)
    # plt.xlabel('f', 0)
    plt.xlabel('Wine Quality Rating')
    plt.ylabel('Instances')
    plt.title('Wine Quality Label Distribution')
    # plt.axes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.show()

    n_bins = 10
    plt.hist(adultY, bins=n_bins)
    plt.xlabel('Wine Quality Binary Label')
    plt.ylabel('Instances')
    plt.title('Wine Quality Split 0 to 5 and 6 to 9')
    # plt.axes([0,1])
    plt.show()


elif(dataset == 'abalone'):
    n_bins = 28
    plt.hist(adultY1, bins=n_bins, color='r')
    plt.xlabel('Abalone Ring Count')
    plt.ylabel('Instances')
    plt.title('Abalone Ring Count Label Distribution')

    plt.show()

    n_bins = 10
    plt.hist(adultY, bins=n_bins, color='r')
    plt.xlabel('Abalone Ring Count Binary Label')
    plt.ylabel('Instances')
    plt.title('Abalone Ring Count Split 1 to 9 and 10 to 29')
    # plt.axes([0,1])
    plt.show()





