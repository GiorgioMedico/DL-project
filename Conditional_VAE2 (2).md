In this notebook we shall present a simple conditional VAE, trained on MNIST


```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```

The conditional autoencoder will allow to generate specific digits in the MNIST range 0-9. The condition is passed as input to encoder and decoder in categorical format.


```python
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
```

# The model

Sampling function for the Variational Autoencoder.
This is the clsed form of the Kullback-Leibler distance between a gaussian N(z_mean,z_var) and a normal prior N(0,1)


```python
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
```

Main dimensions for the model (a simple stack of dense layers).


```python
input_dim = (28,28,1)
latent_dim = 16
```


```python
x = layers.Input(shape=input_dim)
c = layers.Input(shape=(10,))
cv = layers.Lambda(lambda x: tf.expand_dims(x,axis=1))(c)
cv = layers.Lambda(lambda x:tf.expand_dims(x,axis=1))(cv)
cv = layers.UpSampling2D((28,28), interpolation='nearest')(cv)
y = layers.concatenate([x,cv])
y = layers.Conv2D(16,(3,3),strides=(2,2),padding='same')(y)
y = layers.BatchNormalization()(y)
y = layers.Activation('swish')(y)
y = layers.Conv2D(16,(3,3),activation='swish',padding='same')(y)
y = layers.Conv2D(32,(3,3),strides=(2,2),activation='swish',padding='same')(y)
y = layers.Conv2D(32,(3,3),activation='swish',padding='same')(y)
y = layers.Conv2D(64,(3,3),strides=(2,2),activation='swish',padding='same')(y)
y = layers.Conv2D(64,(3,3),activation='swish',padding='same')(y)
y = layers.Flatten()(y)
y = layers.Dense(64,activation='swish')(y)
z_mean = layers.Dense(latent_dim)(y)
z_log_var = layers.Dense(latent_dim)(y)
encoder = Model([x,c],[z_mean,z_log_var])
```

We start with the encoder. It takes two inputs: the image and the category.

It returns the latent encoding (z_mean) and a (log-)variance for each latent variable.


```python
encoder.summary()
```

Now we need to address the decoder. We first define its layers, in order to use them both in the vae model and in the stand-alone generator.

Now we sample around z_mean with the associated variance.

Note the use of the "lambda" layer to transform the sampling function into a keras layer.


```python
x = layers.Input(shape=(latent_dim,))
c = layers.Input(shape=(10,))
y = layers.concatenate([x,c])
y = layers.Dense(128,activation='swish')(y)
y = layers.concatenate([x,c])
y = layers.Dense(1024,activation='swish')(y)
y = layers.Reshape((4,4,64))(y)
y = layers.Conv2D(64,(3,3),activation='swish',padding='same')(y)
y = layers.Conv2DTranspose(32,(3,3),strides=(2,2),activation='swish',padding='same')(y)
y = layers.Conv2D(32,(3,3),activation='swish',padding='same')(y)
y = layers.Conv2DTranspose(16,(3,3),strides=(2,2),activation='swish',padding='same')(y)
y = layers.Conv2D(16,(3,3),activation='swish',padding='valid')(y)
y = layers.Conv2DTranspose(16,(3,3),strides=(2,2),activation='swish',padding='same')(y)
y = layers.Conv2D(16,(3,3),padding='same')(y)
y = layers.BatchNormalization()(y)
y = layers.Activation('swish')(y)
y = layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(y)
decoder = Model([x,c],y)
#decoder = Model(x,y)
```


```python
decoder.summary()
```


```python
class KLDivergenceLayer(layers.Layer):
    """
    Custom Keras layer to calculate the KL divergence loss.

    This layer ensures that the KL divergence calculation is performed
    on concrete tensors during model execution, avoiding the error
    "ValueError: Tried to convert 'x' to a tensor and failed."
    """
    def __init__(self, gamma=0.0001, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Calculate KL divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Apply gamma scaling
        kl_loss = self.gamma * kl_loss
        # Add the KL loss as an activity regularization loss
        self.add_loss(K.mean(kl_loss))
        # Return the original inputs unchanged
        return inputs
```


```python
x = layers.Input(shape=input_dim)
c = layers.Input(shape=(10,))
z_mean, z_log_var = encoder([x,c])
z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_hat = decoder([z,c]) #z
cvae = Model([x,c],x_hat)
```


```python
cvae.summary()
```

We decode the image starting from the latent representation z and its category y, that must be concatenated.

The VAE loss function is just the sum between the reconstruction error (mse or bce) and the KL-divergence, acting as a regularizer of the latent space.


```python
def vae_loss(y_true, y_pred):
    gamma = .0001  #balancing parameter
    # Reconstruction loss
    rec_loss = K.sum(metrics.mse(y_true, y_pred),axis=(1,2))
    # KL divergence loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # Total VAE loss
    total_loss = rec_loss + gamma*kl_loss
    return total_loss
```

Some hyperparameters. Gamma is used to balance loglikelihood and KL-divergence in the loss function


```python
batch_size = 128
epochs = 60
```

We are ready to compile. There is no need to specify the loss function, since we already added it to the model with add_loss.


```python
optimizer = optimizers.Adam(learning_rate=.0005)
```


```python
cvae.compile(optimizer=optimizer,loss='mse')
```

Train for a sufficient amount of epochs. Generation is a more complex task than classification.


```python
cvae.fit([x_train,y_train],x_train,epochs=30,batch_size=batch_size)
```

Let us decode the full test set.


```python
decoded_imgs = cvae.predict([x_test,y_test])
```

The following function is to test the quality of reconstructions (not particularly good, since compression is strong).


```python
def plot(n=10):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
```


```python
plot()
```

Finally, we build a digit generator that can sample from the learned distribution


```python
generator = decoder
```

And we can generate our samples


```python
import time
# display a 2D manifold of the digits
n = 3  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

while True:
  label = input("input digit to generate: \n")
  label = int(label)
  if label < 0 or label > 9:
      print(label)
      break
  label = np.expand_dims(utils.to_categorical(label,10),axis=0)
  for i in range(0,n):
    for j in range (0,n):
        z_sample = np.expand_dims(np.random.normal(size=latent_dim),axis=0)
        x_decoded = generator.predict([z_sample,label])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
  plt.figure(figsize=(5, 5))
  plt.imshow(figure, cmap='Greys_r')
  plt.show()
  time.sleep(1)

```
