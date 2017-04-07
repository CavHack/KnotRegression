import sys
import numpy as np

def load_data(input_file):
    """"
    Load the dataset of interest.
    """"

    data = np.genfromtxt(input_file, delimiter=',', skip_header=0, names=None)
    return data



    def KMeans(X, K=5, maxit=10, saveLog = True):
        """"
       Apply KMeans for clustering a dataset given as input, and the number of clusters (K).
      Input: x1, ..., xn where x in R^d, and K
      Output: Vector c of cluster assignments, and K mean vectors mu
        """"
