# Mnist classification with NNs
A first example of a simple Neural Network, applied to a well known dataset.


```python
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import numpy as np
```

Let us load the mnist dataset


```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
print(x_train.shape)
print("pixel range is [{},{}]".format(np.min(x_train),np.max(x_train)))
```

We normalize the input in the range [0,1]


```python
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train,(60000,28*28))
x_test = np.reshape(x_test,(10000,28*28))
```

Let us visualize a few images


```python
i = np.random.randint(0,60000)
plt.imshow(x_train[i].reshape(28,28),cmap='gray')
```

The output of the network will be a probability distribution over the different categories. Similarly, we generate a ground truth distribution, and the training objective will consist in minimizing their distance (categorical crossentropy). The ground truth distribution is the so called "categorical" distribution: if x has label l, the corresponding categorical distribution has probaility 1 for the category l, and 0 for all the others.


```python
i = np.random.randint(0,60000)
print(y_train[i])
y_train_cat = utils.to_categorical(y_train)
print(y_train_cat[i])
y_test_cat = utils.to_categorical(y_test)
```

Our first Netwok just implements logistic regression


```python
xin = Input(shape=(28*28,))
#x = Dense(64, activation='relu')(xin)
x = Dense(10,activation='softmax')(xin)
#res = Activation('softmax')(x)

mynet = Model(inputs=[xin],outputs=[x])
```


```python
mynet.summary()
```

Now we need to compile the network.
In order to do it, we need to pass two mandatory arguments:


*   the **optimizer**, in charge of governing the details of the backpropagation algorithm
*   the **loss function**

Several predefined optimizers exist, and you should just choose your favourite one. A common choice is Adam, implementing an adaptive lerning rate, with momentum

Optionally, we can specify additional metrics, mostly meant for monitoring the training process.



```python
mynet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```

Finally, we fit the model over the training set.

Fitting, just requires two arguments: training data e ground truth, that is x and y. Additionally we can specify epochs, batch_size, and many additional arguments.

In particular, passing validation data allow the training procedure to measure loss and metrics on the validation set at the end of each epoch.


```python
mynet.fit(x_train,y_train_cat, shuffle=True, epochs=5, batch_size=32,validation_data=(x_test,y_test_cat))
```


```python
mynet.save_weights('myweights.weights.h5')
```


```python
xin = Input(shape=(784,))
x = Dense(47,activation='relu')(xin)
#x = BatchNormalization()(x)
res = Dense(10,activation='softmax')(x)

mynet2 = Model(inputs=xin,outputs=res)
```


```python
mynet2.summary()
```


```python
mynet2.load_weights('myweights')
```


```python
mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```


```python
mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=5, batch_size=32,validation_data=(x_test,y_test_cat))
```


```python
p = mynet.predict(x_test[1:2])
print(p)
print(np.argmax(p))
```


```python
print(y_test[1])
```


```python
mynet.save_weights('myweights')
```

An amazing improvement. WOW!

# Exercises

1.   Add additional Dense layers and check the performance of the network
2.   Replace 'relu' with different activation functions
3. Adapt the network to work with the so called sparse_categorical_crossentropy
4. the fit function return a history of training, with temporal sequences for all different metrics. Make a plot.


