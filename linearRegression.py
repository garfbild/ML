import numpy as np

class linreg():
    def __init__(self,dim):
        self._dim = dim
        self._Theta = np.random.rand(self._dim+1,1)

    def h(self,x):
        return x@self._Theta

    def cost(self,x,y):
        return np.sum(np.square(self.h(x) - y))

m = 10
dim = 1
x = np.zeros((m,dim+1))
y = np.zeros((m,1))
A = 2
B = 1

for i in range(m):
    x[i] = [i,1]
    y[i] = [np.random.normal(A*i + B, 0.1)]

model = linreg(dim)
print(model.cost(x,y))
