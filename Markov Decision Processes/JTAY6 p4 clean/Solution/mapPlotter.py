#################################################################################
# Ryan Lewis
# Code adapted from Jontay
#####################################################################################
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

# import cPickle as pkl

infile = './QL Q-Learning L0.1 q0.0 E0.1 Hard Iter 1000 Policy Map.pkl'
with open(infile,'rb') as f:
    arr = pkl.load(f, encoding='latin1')

lookup = {'None': (0,0),
          '>': (1,0),
        'v': (0,-1),
        '^':(0,1),
        '<':(-1,0)}    

n= len(arr)
arr = np.array(arr)    
X, Y = np.meshgrid(range(1,n+1), range(1,n+1))    
U = X.copy()
V = Y.copy()
for i in range(n):
    for j in range(n):
        U[i,j]=lookup[arr[n-i-1,j]][0]
        V[i,j]=lookup[arr[n-i-1,j]][1]

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V,headaxislength=5,pivot='mid',angles='xy', scale_units='xy', scale=1)
plt.xlim((0,n+1))
plt.ylim((0,n+1))
plt.tight_layout()

plt.show()