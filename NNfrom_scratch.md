In this file we develop a Neural Network from scratch, and implement its backpropagtion algorithm just using mathematical libraries of numpy.

The purpose of the netwoork is to acquire a deeper inside into backpropagation.
The code in this notebook tightly reflects the pseudocode given in the slides.


```
import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
```

Let us define a couple of activation functions (sigmoid and relu) and their derivatives.


```
##############################################
# activation functions
##############################################

def sigmoid(x): return 1 / (1 + math.exp(-x))

def sigderiv(x): return (sigmoid(x)*(1-sigmoid(x)))

def relu(x):
  if x >= 0: return x
  else: return 0

def reluderiv(x):
  if x >= 0: return 1
  else: return 0

def activate(x): return sigmoid(x)  #relu(x)
def actderiv(x): return sigderiv(x) #reluderiv(x)
```

A neural network is just a collection of numerical vectors describing the weigths of the links at each layer. For instance, a dense layer between n input neurons and m output neurons is defined by a matrix w of dimension nxm for the weights and a vector b of dimension m for the biases.

Supposing the network is dense, its architecture is fullly specified by the number of neurons at each layer. For our example, we define a shallow network with 8 input neurons,
3 hidden neurons, and 8 output neurons, hence with dimension [8,3,8].

We initialize weights and biases with random values.


```
##############################################
# net parameters
##############################################

dim = [8,3,8]
l = len(dim)

w,b = [],[]

for i in range(1,l):
  w.append(np.random.rand(dim[i-1],dim[i]))
  b.append(np.random.rand(dim[i]))
```

For the backpropagation algorithm we also need to compute, at each layer, the weighted sum z (inputs to activation), the activation a, and the partial derivative d of the error relative to z.

We define a version of the backpropagation algorithm working "on line", processing a single training sample (x,y) at a time, and updating the nework parameters at each iteration. The backpropagation function also return the current error  relative to (x,y).

An epoch, is a full pass of the error update on all training data; it returns the cumulative error on all data.


```
##############################################
# training - on line, one input data at a time
##############################################

mu = 1

z,a,d=[],[],[]

for i in range(0,l):
  a.append(np.zeros(dim[i]))

for i in range(1,l):
  z.append(np.zeros(dim[i]))
  d.append(np.zeros(dim[i]))

def update(x,y):
  #input
  a[0] = x
  #feed forward
  for i in range(0,l-1):
    z[i] = np.dot(a[i],w[i])+b[i]
    a[i+1] = np.vectorize(activate)(z[i])
  #output error
  d[l-2] = (y - a[l-1])*np.vectorize(actderiv)(z[l-2])
  #back propagation
  for i in range(l-3,-1,-1):
    d[i]=np.dot(w[i+1],d[i+1])*np.vectorize(actderiv)(z[i])
  #updating
  for i in range(0,l-1):
    for k in range (0,dim[i+1]):
      for j in range (0,dim[i]):
        w[i][j,k] = w[i][j,k] + mu*a[i][j]*d[i][k]
      b[i][k] = b[i][k] + mu*d[i][k]
    if False:
      print("d[%i] = %s" % (i,(d[i],)))
      print("b[%i] = %s" % (i,(b[i],)))
  #print("error = {}".format(np.sum((y-a[l-1])**2)))
  return np.sum((y-a[l-1])**2)

def epoch(data):
    e = 0
    for (x,y) in data:
      e += update(x,y)
    return e
```

Now we define same data and fit the network over them.

We want to define a simple example of autoencoder, taking in input a one-hot representation of the numbers between 0 and 7, and trying to compress them to a
boolean internal representation on 3 bits.



```
X = [[1,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,0,0,1,0,0,0],
     [0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,1]]

def data(): return zip(X,X)

final_error = .003
dist = epoch(data())

while dist > final_error:
  print("distance= %f" % dist)
  dist = epoch(data())

print("distance= %f" % dist)
for x in X:
  print("input = %s" % (x,))
  a[0] = x
  #feed forward
  for i in range(0,l-2):
    z[i] = np.dot(a[i],w[i])+b[i]
    a[i+1] = np.vectorize(activate)(z[i])
  print("hidden level = %s" % (a[i+1],))
  z[l-2] = np.dot(a[l-2],w[l-2])+b[l-2]
  a[l-1] = np.vectorize(activate)(z[l-2])
  #print("output = %s" % (a[l-1],))
```

You should interpret the latent representation as a binary encoding:




```
X = [[1,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,0,0,1,0,0,0],
     [0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,1]]

def data(): return zip(X,X)

final_error = .003
dist = epoch(data())

while dist > final_error:
  print("distance= %f" % dist)
  dist = epoch(data())

print("distance= %f" % dist)
for x in X:
  print("input = %s" % (x,))
  a[0] = x
  #feed forward
  for i in range(0,l-2):
    z[i] = np.dot(a[i],w[i])+b[i]
    a[i+1] = np.vectorize(activate)(z[i])
  print("hidden level = %s" % (a[i+1],))
  z[l-2] = np.dot(a[l-2],w[l-2])+b[l-2]
  a[l-1] = np.vectorize(activate)(z[l-2])
  #print("output = %s" % (a[l-1],))
```

You may interpret the latent representation as a sort of binat encoding: the network is learning binarization!

Latent [0.01 0.01 0.01]  ->  0 0 0

Latent [0.99 0.99 1.00]    ->  1 1 1

Latent [0.00   0.98 0.98]  ->  0 1 1

Latent [0.01 0.96 0.01]  ->  0 1 0

Latent [0.98 0.99 0.01]  ->  1 1 0

Latent [0.98 0.00   0.97]  ->  1 0 1

Latent [0.97 0.02 0.01]  ->  1 0 0

Latent [0.01 0.01 0.91]  ->  0 0 1

Exercises.

1.   change the specification of the network to allow a different activation function for each layer;
2.   modify the backpropagation algorithm to work on a minibatch of samples.




