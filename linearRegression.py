import matplotlib.pyplot as plt
import numpy as np


class linreg():
    def __init__(self,dim):
        self._dim = dim
        self._Theta = np.random.rand(self._dim+1,1)

    def h(self,x):
        return x@self._Theta

    def cost(self,x,y):
        return np.square(self.h(x) - y)/2

    def J(self,x,y):
        m = y.shape[0]
        return np.sum(self.cost(x,y))/m

    def dJ(self,x,y):
        return (np.transpose(x)@(self.h(x)-y))/(y.shape[0])

    def gradientDescent(self,alpha,max_iter,x,y):
        m = y.shape[0]
        for n in range(max_iter):
            self._Theta = self._Theta - alpha*self.dJ(x,y)/m
            print(self.J(x,y))

    def graph(self):
        x = np.random.rand(5,self._dim+1)
        for i in range(5):
            x[i,-1] = 1
        y = self.h(x)
        plt.scatter(x[:,:-1],y)
        plt.show()

m = 10
dim = 1
x = np.zeros((m,dim+1))
y = np.zeros((m,1))
A = 2
B = 1

for i in range(m):
    x[i] = [i,1]
    y[i] = [np.random.normal(A*i + B, 0.2)]

model = linreg(dim)
model.gradientDescent(0.1,10000,x,y)
#model.graph()
print(model._Theta)
