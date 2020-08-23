#https://www.youtube.com/watch?v=TS32rBteHh4
#https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Calculation

import numpy as np
import inspect

def absMax(x):
    posMax = x.max()
    negMax = x.min()
    if posMax >= -negMax:
        return posMax
    else:
        return negMax

def EigenFunction(A,max_iter=1000):
    x = np.ones((A.shape[0],1))*2
    for k in range(max_iter):
        x = A@x
        x = x /abs(x)
    s = (A@x)/x
    return x,s[0]

x = np.random.normal([0,1],[1,2],(3,2))
m = x.shape[0]

mu = np.sum(x,axis = 0)/m
sigma = np.sqrt(np.sum(np.square(x-mu),axis = 0)/m)

x = (x-mu)/sigma

Covariance = (np.transpose(x)@x)/m

A = np.array([[2,1],[1,2]])
print(EigenFunction(A))
