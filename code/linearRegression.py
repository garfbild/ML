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
        return np.sum(self.cost(x,y))+np.sum(self._Theta)/m

    def dJ(self,x,y):
        return np.transpose(x)@(self.h(x)-y)

    def gradientDescent(self,x,y,Alpha,max_iter):
        m = y.shape[0]
        for n in range(max_iter):
            self._Theta = self._Theta - Alpha*self.dJ(x,y)/m

    def gradientDescentRegularised(self,x,y,Alpha,Lambda,max_iter):
        m = y.shape[0]
        for n in range(max_iter):
            self._Theta = self._Theta*(1-Lambda*Alpha/m) - Alpha*self.dJ(x,y)/m

    def graph(self):
        x = np.random.rand(5,self._dim+1)
        for i in range(5):
            x[i,-1] = 1
        y = self.h(x)
        plt.scatter(x[:,:-1],y)
        plt.show()

m = 100000
dim = 1
x = np.zeros((m,dim+1))
y = np.zeros((m,1))
A = 2
B = 1

for i in range(m):
    x[i] = [(np.random.random()*20)-10,1]
    y[i] = [np.random.normal(A*x[i,0] + B, 0.2)]

model = linreg(dim)
model.gradientDescentRegularised(x,y,0.02,1,1000)
#model.graph()
print(model._Theta)
