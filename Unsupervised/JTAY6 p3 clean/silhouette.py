from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys


print(__doc__)

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


X = abaloneX
Y = abaloneY
#
X = wineX
Y = wineY

range_n_clusters = [2, 3, 4, 5, 6]
range_n_clusters=  [2,3,4,5,10,15,20,25,30,35,40]
# range_n_clusters = [2]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # ax1.set_title("The silhouette plot for the various clusters.")
    # ax1.set_xlabel("The silhouette coefficient values")
    # ax1.set_ylabel("Cluster label")
    #
    # # The vertical line for average silhouette score of all the values
    # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    #
    # ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #
    # # 2nd Plot showing the actual clusters formed
    # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')
    #
    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')
    #
    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')
    #
    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")
    #
    # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
    #               "with n_clusters = %d" % n_clusters),
    #              fontsize=14, fontweight='bold')

    # plt.show()