##############################################################
Author: Ryan Lewis with code adapted from Jonathan Tay
Date: 9/18/2017
##############################################################

#############################################################################
The assignment code is written in Python 3.6.1. Library dependencies are:
scikit-learn 0.18.1
numpy 1.12.1
pandas 0.20.1
matplotlib 2.0.2

To run the code you will want to run any of the algorithm named files in the JTAY p1 clean/ folder.
The output excel files will appear in JTAY p1 clean/output/

#############################################################################


#############################################################################
The main folder contains the following files:
1. winequality-white.csv, abalone.data -> These are the original datasets, as downloaded from the UCI Machine Learning Repository http://archive.ics.uci.edu/ml/
3. readme.txt -> This file
4. DTreePruning.py -> File to implement post pruning
5. KNN.py -> plot of validation curve
6. plotter.py -> experimental plotting file

JTAY6 p1 clean folder contains:
1. helpers.py -> A collection of helper functions used for this assignment
2. ANN.py -> Code for the Neural Network Experiments
3. Boosting.py -> Code for the Boosted Tree experiments
4. "Decision Tree.py" -> Code for the Decision Tree experiments
5. KNN.py -> Code for the K-nearest Neighbours experiments
6. SVM.py -> Code for the Support Vector Machine (SVM) experiments


There is also a subfolder /JTAY6 p1 clean/output. This folder contains the results.
Here, I use DT/ANN/BT/KNN/SVM_Lin/SVM_RBF to refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbours, linear and RBF kernel SVMs respectively.
There files in the output folder. They come the following types:
1. <Algorithm>_<dataset>_reg.csv -> The validation curve tests for <algorithm> on <dataset>
2. <Algorithn>_<dataset>_LC_train.scv -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. Used for learning curves.
3. <Algorithn>_<dataset>_LC_test.csv -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. Used for learning curves.
4. <Algorithm>_<dataset>_timing.csv -> Table of fraction of training set vs. training and evaluation times. If the fulll training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
5. ITER_base_<Algorithm>_<dataset>.csv -> Table of results for learning curves based on number of iterations/epochs.
6. ITERtestSET_<Algorithm>_<dataset>.csv -> Table showing training and test set accuracy as number of iterations/epochs is varied. NOT USED in report.
7. "test results.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.
##########################################################################################################################################################