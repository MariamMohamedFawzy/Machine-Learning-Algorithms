import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import randint


# from http://www.codehamster.com/2015/03/09/different-ways-to-calculate-the-euclidean-distance-in-python/
# start
def euclidean0_1(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist
# end

def calc_distances(X, C):
    D = []
    for x in X:
        d = []
        for c in C:
            d_new = euclidean0_1(x, c)
            d.append(d_new)
        D.append(min(d))
    return D

def init_clusters_centers(X, K):
    start = randint(0,len(X)-1)
    C = np.array([X[start, :]])
    for k in range(2, K+1):
        D = calc_distances(X, C)
        max_elm = D.index(max(D))
        temp = np.array([X[max_elm, :]])
        C = np.concatenate((C, temp), axis=0)
    return C    

# Importing the dataset
datasetPath = '/Users/apple/Desktop/Data science/courses/Machine_Learning_A-Z_Hands-On_Python_and_R_In_Data_Science/21 K-Means Clustering/attached_files/115 K-Means Clustering in Python/K_Means/Mall_Customers.csv'
dataset = pd.read_csv(datasetPath)
X = dataset.iloc[:, [3, 4]].values

C = init_clusters_centers(X, 5)

plt.scatter(X[:, 0], X[:, 1], c='b')
plt.scatter(C[:, 0], C[:, 1], c='r')
plt.show()
    
