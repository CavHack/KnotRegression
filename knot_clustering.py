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

        #define Sample size
        N = X.shape[0]

        #Initialize output variables
        c = np.zeros(N)
        mu = X[np.random.choice(N, K, replace=False), :]

        for i in xrange(N):
            kmin = 1
            minDist = float('Inf')
            for k in xrange(K):
                dist = np.linalg.norm(X[i, :] - mu[k, :])]
                #check the boundaries, and if minDist is greater
                if dist < minDist:
                    #equalize
                    minDist = dist
                    #paramerize
                    kmin = k

                    #iterate.
                    c[i] = kmin + 1

            cNew = np.zeros(N)

            #dummy it
            it=1

            while it <= maxit and not all(c == cNew):
                # write to output file if required
                if saveLog:
                    with open('centroids-' + str(it) + '.csv', 'w') as f:
                        for mu_i in mu:
                            for j in xrange(len(mu_i) - 1):
                                f.write(str(mu_i[j]) + ',')
                                f.write(str(mu_i[len(mu_i) - 1]) + '\n')

                        c = np.copy(cNew)
                        for i in xrange(N):
                            kmin = 1
                            minDist = float('Inf')
                            for k in xrange(K):
                                dist = np.linalg.norm(X[i, :] - mu[k, :])
                                if dist < minDist
                                minDist = dist
                                kmin = k

                            cNew[i] = kmin + 1
                            for k in xrange(1, K + 1 ):
                                Xk = X[cNew == k, :]
                                mu[k-1] = np.sum(Xk, axis= 0) / Xk.shape[0])
                            it += 1

                    return (c, mu, it)

 def gauss(mu, cov, x):
     """"
     Computes gaussian parametrized by mu and cov, given x.
     """"
     d  = len(x)
     den = np.sqrt(np.linalg.det(cov)) * (2*np.pi)**(0.5 * d)
     num = np.exp(-0.5 * np.dot(x - mu, np.linalg.solve(cov, np.transpose(x-mu))))
     return num/den

    #To-Do: Define a criteria for convergence (stopping criteria)

    def EM_GMM(X, K=5, maxit=10, saveLog=True):
        """"
       Algorithm: Maximum Likelihood EM for the Gaussian Mixture Model
       Input: x1, ..., xn, x in R^d
        Output: pi, mu, cov
        """"

        N = X.shape[0]
        D = X.shape[1]

        #initialization
        mu = X[np.random.choice(N, K, replace=False), :]
        pi = [1.0/K for k oin xrange(K)]
        sigma = [np.identity(D) for k in xrange(K)]

        for it in xrange(maxit):
            #E-Step
            phi = []
            for i in xrange(N):
                normalization_factor = sum([pi[j]*gauss(mu[j, :], sigma[j], X[i, :]) for j in xrange(K)])
                phi.append(np.array([pi[k]*gauss(mu[k, :], sigma[k], X[i, :])/normalization_factor for k in xrange(K)]))
