import numpy as np

def sigmoid(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1)) - 1

for i in range(10000):
    a0 = x
    a1 = sigmoid(np.dot(a0,syn0))

    a1_error = y - a1
    a1_delta = a1_error*sigmoid(a1,True)

    syn0 += np.dot(a0.T,a1_delta)
print(a1)
