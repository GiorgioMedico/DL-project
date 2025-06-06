# Overfitting on the IMDB movie reviews dataset

In this notebook we shall focus on overfitting, demonstrating the phenomenon and studying techniques to address it. The dataset we shall use is the IMDB movie reviews dataset, composed of 25,000 movies reviews, labeled by sentiment (positive/negative).

To prevent overfitting, the best solution is to use more training data. When that is not a viable possibility, you can try to use regularization techniques, constraining the quantity and quality of information stored by the model. If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent ones, which have a better chance of generalizing well.




```
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np
```

Let us load the dataset. In keras, the dataset is preprocessed, and each review is encoded as a sequence of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data.


```
num_words = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=num_words)
```

Let us a look at the encoding of the first review.




```
print(x_train[0])
```

This representation has a variable length dimension, that is not very stuitable for a neural network.

Let us transform it into a multi_hot encoding of of dimension equal to num_words. In this representation, a word gets index 1 if it appears in the document. It is essentially a bag-of-words encoding.

"this film is very bad" -> 0
"this film is not so bad" -> 1





```
def multi_hot_sequences(sequences, dimension):
  multi_hot = np.zeros((len(sequences),dimension))
  for i in range(0,len(sequences)):
    multi_hot[i, sequences[i]] = 1
  return multi_hot


x_train = multi_hot_sequences(x_train, num_words)
x_test = multi_hot_sequences(x_test, num_words)
```

Let us have a look at the initial part of the encoding for the first review.


```
print(x_train[0,0:30])

print( x_test.shape)
```

We now define our first model, that is just a concatenation of three dense layers.


```
seq = Input(shape=(num_words,))
x = Dense(64, activation='relu')(seq)
x = Dense(16, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

base_model = Model(seq, x)
```


```
base_model.summary()
```

We compile the model using adam as optimizer, and binary crossentropy (log likelyhood) as loss function. The fit function returns a history of the training, that can be later inspected. In addition to the loss function, that is the canonical metric used for training, we also ask the model to keep trace of accuracy.


```
base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```
base_history = base_model.fit(
    x_train,
    y_train,
    epochs = 6,
    batch_size = 512,
    validation_data = (x_test, y_test),
    verbose = 1
)
```

Let us see the keys available in our history.


```
print(base_history.history.keys())
```

The following function allows us to plot the results.


```
def plot_history(model_history,keys):
    m,val_m = keys
    plt.plot(model_history.history[m])
    plt.plot(model_history.history[val_m])
    plt.ylabel(m)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

plot_history(base_history,['accuracy','val_accuracy'])
plot_history(base_history,['loss','val_loss'])

```

# Weight regularization

Now we modify our base model adding regularizers.

A common way to mitigate overfitting is to reduce the complexity of the network by forcing its weights to only take small values, making the distribution of weights more “regular”. This is called “weight regularization”, and it is done by adding to the loss function of the network an additional cost associated with having large weights.


```
from keras import regularizers

seq = Input(shape=(num_words,))
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(seq)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
x = Dense(1, activation='sigmoid')(x)

l2reg_model = Model(seq, x)
```


```
l2reg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```
l2reg_history = l2reg_model.fit(
    x_train,
    y_train,
    epochs = 15,
    batch_size = 512,
    validation_data = (x_test, y_test),
    verbose = 2
)
```


```
plot_history(l2reg_history,['accuracy','val_accuracy'])
plot_history(l2reg_history,['loss','val_loss'])
```

# Dropout

Dropout is an alternativeregularization techniques for neural networks. It consists of randomly “dropping out” (i.e. set to zero) a number of output features of the layer during training.

At test time, no units are dropped out, but the layer’s output values are scaled down by a factor equal to the dropout rate, so as to balance for the fact that more units are active than at training time.


```
from keras.layers import Dropout
from keras import optimizers
```

Let’s add a couple of dropout layers in our IMDB network and see how it performs.



```
seq = Input(shape=(num_words,))
x = Dense(64, activation='relu')(seq)
x = Dropout(0.5)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

dropout_model = Model(seq, x)
```


```
dropout_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```
dropout_model.summary()
```


```
dropout_history = dropout_model.fit(
    x_train,
    y_train,
    epochs = 5,
    batch_size = 512,
    validation_data = (x_test, y_test),
    verbose = 1
)
```


```
plot_history(dropout_history,['accuracy','val_accuracy'])
plot_history(dropout_history,['loss','val_loss'])
```


```
seq = Input(shape=(num_words,))#bs,500
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(seq)
x = Dropout(0.5)(x)
x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

mixed_model = Model(seq, x)
```


```
adam = optimizers.Adam(learning_rate=0.001)
mixed_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
```


```
mixed_history = mixed_model.fit(
    x_train,
    y_train,
    epochs = 5,
    batch_size = 512,
    validation_data = (x_test, y_test),
    verbose = 2
)
```


```
plot_history(mixed_history,['accuracy','val_accuracy'])
plot_history(mixed_history,['loss','val_loss'])
```

# Early stopping

Early stopping is a method that allows you to stop training as soon as the model performance stops improving on the validation dataset.

This requires that a validation set must be provided to the fit() function.

Early stopping can be simply implemented in keras using callbacks.
A callback is a function taht is called at specific stages of the training procedure: start/end of epochs, start end of minibatches, etc.

You can use callbacks to get a view on internal states and statistics of the model during training. A list of callbacks can be passed to the .fit() function using the keyword argument "callbacks".


```
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

saveDir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

chkpt = os.path.join(saveDir, 'Cifar10_to256.keras')

mixed_model.load_weights(chkpt)

es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
```


```
mixed_model.fit(x_train, y_train,
                batch_size=512, #batch_size,
                epochs= 20,
                verbose=2,
                validation_data=(x_test,y_test),
                callbacks=[es_cb, cp_cb],
                shuffle=True)
```


```
loss,acc = mixed_model.evaluate(x_test,y_test)
print("test loss = ", loss)
print("test accuracy = ", acc)
```
