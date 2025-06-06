The purpose of this notebook is to show the use of LSTMs for processing sequences. 

Specifically, we try to compute the sum of two binay digits,
delegating to the model the task of taking care of the propagation of the carry.


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Lambda
from tensorflow.keras.models import Model
from 
```

Here is our generator. Each element of the resulting batch is a pair (a,res)
where a[0] and a[1] are two sequences of lenght seqlen of binary digits, and
res is their sum. The digits are supposed to be represented in a positional order with less significative digits at lower positions (left to rigth).

The initial carry of the generator is 0; at successive invocations it 
reuses the final carry of the previous sum.


```python
def generator(batchsize,seqlen):
    init_carry = np.zeros(batchsize)
    carry = init_carry
    while True:
      #print("initial carry = ", carry)
      a = np.random.randint(2,size=(batchsize,seqlen,2))
      res = np.zeros((batchsize,seqlen))
      for t in range(0,seqlen):
        sum = a[:,t,0]+a[:,t,1] + carry
        res[:,t] = sum % 2
        carry = sum // 2
      yield (a, res)
```

Let's create an instance of the generator.


```python
gen = generator(1,2)
```

And now let's see a few samples.


```python
a,res = next(gen)
print("a1 = {}, a2={}. res = {}".format(a[0,:,0],a[0,:,1],res[0]))
```

We can now define the model. It takes in input a pair of boolean sequences of unspecified length. The batchsize dimension is, as usual, implicit too.


```python
def gen_model():
    xa = Input(shape=(None,2))
    x = Conv1D(8,1,activation='relu')(xa)
    x = Conv1D(4,1,activation='relu')(x)
    #x = xa
    x = LSTM(4,activation=None, return_sequences=True)(x)
    x = Dense(1,activation='sigmoid')(x)
    out = tf.squeeze(x,2)
    #out = x
    comp = Model(inputs=xa, outputs=out)
    return comp
```


```python
mymodel = gen_model()
mymodel.summary()
```


```python
mymodel.compile(optimizer='adam',loss='mse')
```


```python
batchsize=100
seqlen=10
```


```python
#mymodel.load_weights("weights/lstm.hdf5")
```


```python
mymodel.fit(generator(batchsize,seqlen), steps_per_epoch=100, epochs=100)
#comp.save_weights("weights/lstm.hdf5")
```


```python
example,res = next(generator(1,10))
predicted = np.array([int(np.rint(x)) for x in mymodel.predict(example)[0]])

print("a1        = {}".format(example[0][:,0].astype(int)))
print("a2        = {}".format(example[0][:,1].astype(int)))
print("expected  = {}".format(res[0].astype(int)))
print("predicted = {}".format(predicted))
```

WOW!
