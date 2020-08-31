#https://www.youtube.com/watch?v=TS32rBteHh4
#https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Calculation

import numpy as np
import matplotlib.pyplot as plt
import inspect

def GershgorinFunction(A):
    centre = []
    radius = []
    for i in range(A.shape[0]):
        centre.append(A[i,i])
        radius.append(np.sum(A[i,:]) - A[i,i])

    return centre, radius

def EigenPowerFunction(A,max_iter=1000):
    x = np.ones((A.shape[0],1))
    for k in range(max_iter):
        x = A@x
        x = x /np.max(abs(x))
    s = (A@x)/x
    return x,round(s[0],5)

def EigenShiftedPowerFunction(A,alpha,max_iter=1000):
    I = np.identity(A.shape[0])
    sA = np.linalg.inv(A - alpha*I)
    x = np.random.rand(A.shape[0],1)
    x = x/np.linalg.norm(x)
    for k in range(max_iter):
        x = sA@x
        x = x/np.linalg.norm(x)
    w = (sA@x)/x
    s = (1/w[0]) + alpha
    return x,round(s[0],5)

# if not any(((e==EigenVector).all() for e in EigenVectorList)) and not any(((-1*e==EigenVector).all() for e in EigenVectorList)):
#     EigenVectorList.append(EigenVector)
# EigenVector = EigenShiftedPowerFunction(A,centre[i]-radius[i]/2)
# if not any(((e==EigenVector).all() for e in EigenVectorList)) and not any(((-1*e==EigenVector).all() for e in EigenVectorList)):
#     EigenVectorList.append(EigenVector)
#spent a lot of time on this when I could have justr checked whetehr the corresponding eigen value had been found instead

def EigenFunction(A):
    centre, radius = GershgorinFunction(A)
    EigenVectorList = []
    EigenValueList = []
    for i in range(A.shape[0]):
        EigenVector,EigenValue = EigenShiftedPowerFunction(A,centre[i]+radius[i]/2)
        if EigenValue not in EigenValueList:
            EigenVectorList.append(EigenVector)
            EigenValueList.append(EigenValue)

        EigenVector,EigenValue = EigenShiftedPowerFunction(A,centre[i]-radius[i]/2)
        if EigenValue not in EigenValueList:
            EigenVectorList.append(EigenVector)
            EigenValueList.append(EigenValue)
    return EigenVectorList, EigenValueList

x = np.random.multivariate_normal([1,1],[[5,3],[3,10]],1000)
m = x.shape[0]

mu = np.sum(x,axis = 0)/m
# sigma = np.sqrt(np.sum(np.square(x-mu),axis = 0)/m)
x = (x-mu)
Covariance = (np.transpose(x)@x)/m

v, s = EigenFunction(Covariance)
maxComponent = s.index(max(s))

plt.hist(x@v[maxComponent])
plt.show()

gradient = v[maxComponent][1]/v[maxComponent][0]

linspace = np.linspace(-5,5,100)
f = [gradient*i for i in linspace]
plt.plot(linspace,f,color = 'r')
plt.scatter(x[:,0],x[:,1])
plt.show()
