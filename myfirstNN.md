This is a very simple example of neural network. Its purpose is to approximate an unknown single valued function using a dense deep network.
The user is invited to play with it, modifying:
1. the hidden function
2. the number of layers and neurons per layer
3. activation functions, number of epochs, and so on.


```
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
```

"myhiddenfunction" is the definition of the function you want to approximate by means of a neural network (NN). The definition is hidden to the NN, who can only access it as a blackbox, to get training samples.
This is implemented by means of a generator (a special kind of function), taking in input a number (batchsize) and returning a pair of input output vectors of length batchsize. Each input is a random number in the interval [-pi,pi] and the output is computed by means of myhiddenfunction.


```
def myhiddenfunction(x):
  #define your favourite function
  #output in range 0-1 if last activation is a sigmoid!
  res = (np.sin(x)**2 + np.cos(x)/3 + 1)/3
  #res = 0.2 + 0.4*x**2 + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)
  #if x < -1 or x > 1 : res = 0
  #else: res = 1
  return res
```

If you have a way to define new samples, you should take advantage of it, exploiting a **generator**.

In python, a generator is similar to a normal function but with a yield statement instead of a return statement.

The difference is that while a "return" statement terminates a function entirely, a "yield" statement pauses the function saving its state and resume from it on successive calls.


```
def generator(batchsize):
    while True:
      #adjust range according to myhiddentfunction
      inputs = np.random.uniform(low=-np.pi,high=np.pi,size=batchsize)
      #inputs = np.random.uniform(low=0,high=1,size=batchsize)
      outputs = np.zeros(batchsize)
      for i in range(0,batchsize):
          outputs[i] = myhiddenfunction(inputs[i]) # + np.random.normal() *.1
      yield (inputs,outputs)
```

If you want to see an example of a generated batch, you need to invoke next on the generator


```
print(next(generator(3)))
```

As we shall see, we may directly call the generator during the training process.

However, for the moment we use it to create a fixed training set.


```
x_train, y_train = next(generator(6000))
x_val, y_val = next(generator(1000))
```

Now we define the network. The function we want to approximate is single valued, so the network will have a single input and a single output, and its (dense) structure is completely defined by a
list specifying the number of neurons per layer



```
input_layer = Input(shape=(1,))
x = Dense(50,activation='relu')(input_layer)   #20 #50 #100
x = Dense(20, activation='relu')(x)
#x = Dense(50, activation='relu')(x)
output_layer = Dense(1,activation='sigmoid')(x)

mymodel = Model(input_layer,output_layer)
```

We can now have a look at the model we just generated:


```
mymodel.summary()
```

Try to be sure you correctly understand the number of learning parameters for each layer.
For a dense layer with n input neurons and m output neurons, your have nxm weights + m biases.
For instance, for a 20-to-30 layer, we have 20x30+30 = 630 parameters.

We are finally ready to compile our model and train it.
As loss function we use mean square error (mse).
The "optimizer" is the technique used to tune the learning rate during backpropagation: you may ignore it for the moment.


```
mymodel.compile(optimizer='adam', loss='mse')
```


```
batchsize = 128
mygen = generator(batchsize)
#mymodel.fit(x_train,y_train,epochs=100,batch_size=batchsize,validation_data=(x_val,y_val))
mymodel.fit(mygen,epochs=50,steps_per_epoch=100)
```


```
def plot(f, model):
  x = np.arange(-np.pi,np.pi,0.05)
  #x = np.arange(0,1,0.005)
  y = [f(a) for a in x]
  z = model.predict(np.array(x))
  plt.plot(x,y,'r',x,z,'b')
  plt.show()
```


```
plot(myhiddenfunction,mymodel)
```


```
batchsize = 64

mymodel.fit(generator(batchsize), steps_per_epoch=1000, epochs=10)
```

If everything is working correctly, the loss should decrease during training.  
If it doesn't, it means that, for some reason, the network is not learning.

We are finally ready to check the result of the approximation. We plot the hidden function in red,
and the approximation computed by the network in blu.


```
x = np.arange(-np.pi,np.pi,0.05)
y = [myhiddenfunction(a) for a in x]
z = mymodel.predict(np.array(x))
plt.plot(x,y,'r',x,z,'b')
plt.show()

```

Now is your turn. Modify:

1. the definition of the hidden function
2. the number of layers/neurons per layer; you just have to change inner_layers_dims in block 6.

Have fun.
