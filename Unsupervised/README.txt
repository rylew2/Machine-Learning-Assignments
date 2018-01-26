#################################################################################
# Ryan Lewis
# Code adapted from Jontay
#####################################################################################

Code Files:
-------------------------------------------------------------------------------------------
benchmark.py - Code to establish performance benchmark of neural network on these datasets
helpers.py - Miscellaneous helper functions
clustering.py - Code for Clustering experiments
PCA.py, ICA.py, RP.py, RF.py - Code for PCA, ICA, Random Projections and Random Forest Feature Selection respectively
silhouette.py - Code to run silhouette score and plot

There are also a number of folders:
---------------------------------------
BASE - Output folder for clustering on the original features
PCA - Output folder for experiments with PCA
ICA - Output folder for experiments with ICA
RP  - Output folder for experiments with Random Projections
RF - Output folder for experiments with Random Forest Feature Selection


To run the experiments:
-------------------------
Generate the data files from the original data by running the parse.py code. This will also generate the appropriate directory structure for the rest of the experiments.
Run clustering.py at the command line with the argument "BASE". eg: python clustering.py BASE
Run each of the scripts ICA, PCA, RP and RF in turn
Run clustering.py at the command line with the arguments ICA, PCA, RP or RF depending on the desired result set. The run.bat file will do this.

Within the output folders, the data files are csv files. They are labelled by dataset and experiment type.


