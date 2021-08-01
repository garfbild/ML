import numpy as np

class markovChain:
    def __init__(self,n):
        self._matrix = np.zeros([2,2])

M = markovChain(3)
print(M._matrix)
