import numpy as np
import math

def Sigmoid(x):
    return np.divide(1,1+np.exp(-1*x))

def SigmoidPrime(x):
    return np.multiply(Sigmoid(x),(np.subtract(1,Sigmoid(x))))

class DenseLayer():
    def __init__(self,InputDim,Nodes):
        self.Theta = np.random.rand(InputDim,Nodes)

    def ComputeForwardPass(self,x):
        z = np.dot(x,self.Theta)
        a = Sigmoid(z)
        return z,a

class Model():
    def __init__(self):
        self.layers = []
        self.a = []
        self.z = []
        self.deltas = []

    def add(self,layer):
        self.layers.append(layer)

    def ForwardPass(self,x):
        self.a = [x]
        self.z = []
        a = x
        for i in range(len(self.layers)):
            z,a = self.layers[i].ComputeForwardPass(a)
            self.z.append(z)
            self.a.append(a)
        return a

    def BackPropagate(self,x,y):
        #output layer
        d3 = self.a[-1]-y
        d2 = self.layers[1].Theta@d3*SigmoidPrime(self.z[0])
        print(d3.shape)
        print(d2.shape)
        print(self.a)
        D1 = d2*self.a[0]
        D2 = d3*self.a[1]

        print(D1.shape,D2.shape)


model = Model()
model.add(DenseLayer(2,4))
model.add(DenseLayer(4,1))
model.ForwardPass(np.array([1,2]))
model.BackPropagate(np.array([1,2]),np.array([1]))
