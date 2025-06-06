# Satellite image inpainting

The project concerns image inpainting, which consists in filling in damaged or missing parts of an image to reconstruct a complete image.

The dataset considered is the EuroSAT Tensorflow dataset based on Sentinel-2 satellite images, in the rgb version. This includes 27000 images, at 64x64 resolution.

A portion of the image is randomly masked according to the procedure described below. The goal is to reconstruct the complete image.




```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

###Dataset



```python
ds_train, ds_info = tfds.load(
    'eurosat/rgb',
    shuffle_files=False,
    #as_supervised=True,  # Returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}
    with_info=True
)
```


```python
ds_train = ds_train['train']
```


```python
ds_train = ds_train.shuffle(1000, seed = 42)
```


```python
train_dataset = ds_train.take(20000)
test_dataset = ds_train.skip(20000)
```

## Generator


The generator provides your training data. We create a mask by drawing random vertical lines at different angles and with varying widths. The portion of the image that is preserved is the part under the mask.



```python
def generator(dataset,nolines=9):
    while True:  # Start an infinite loop
        for batch in dataset:
            images = batch["image"]
            images_np = images.numpy()

            masks = np.zeros((batch_size, 64, 64))
            for i in range(batch_size):
                for j in range(nolines):
                    start_point = (np.random.randint(0, 64 - 1), 0)
                    end_point = (np.random.randint(0, 64 - 1), 63)
                    thickness = np.random.randint(2, 3)
                    masks[i] = cv2.line(masks[i], start_point, end_point, (1), thickness)

            images_np = images_np / 255.0
            masks = np.stack(((masks),) * 3, axis=-1)

            yield (images_np * masks, images_np)

# Batch the datasets
batch_size = 100
train_dataset_batched = train_dataset.batch(batch_size)
test_dataset_batched = test_dataset.batch(batch_size)

# Create generators for the batched datasets
train_generator = generator(train_dataset_batched)
test_generator = generator(test_dataset_batched)
```

Let's visualize the data. In the first row we show the damaged images, and in the second the originals that need to be reconstructed.



```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  # Adjust figsize as needed
a,b = next(train_generator)
for i in range(3):
  # Plot image on each subplot
  axes[0,i].imshow(a[i])  # Use cmap='gray' if your images are grayscale
  axes[0,i].axis('off')  # Turn off axis
  axes[1,i].imshow(b[i])  # Use cmap='gray' if your images are grayscale
  axes[1,i].axis('off')  # Turn off axis

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()
```


```python
from tensorflow import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
```

## U-net


```python
def down_block(inputs, filters, kernel_size=(3, 3), padding='same', activation='relu', dropout=0.5):
  conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
  conv = BatchNormalization()(conv)
  conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
  pool = MaxPooling2D(pool_size=(2, 2))(conv)
  pool = BatchNormalization()(pool)
  pool = Dropout(dropout)(pool)
  return conv, pool

def up_block(inputs, skip, filters, kernel_size=(3, 3), padding='same', activation='relu',dropout=0.5):
  up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
  concat = concatenate([up, skip], axis=3)
  conv = Dropout(dropout)(concat)
  conv = Conv2D(filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
  conv = BatchNormalization()(conv)
  conv = Conv2D(filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
  conv = BatchNormalization()(conv)
  return conv
```


```python
def nn():
  inputs = Input((64, 64, 3))
  # Down Blocks
  conv1, pool1 = down_block(inputs, filters=64, dropout=0.25)
  conv2, pool2 = down_block(pool1, filters=128)
  conv3, pool3 = down_block(pool2, filters=256)

  # bottleneck
  conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

  # Up Blocks
  conv5 = up_block(conv4, skip=conv3, filters=256)
  conv6 = up_block(conv5, skip=conv2, filters=128)
  conv7 = up_block(conv6, skip=conv1, filters=64)

  conv8 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv7)
  return Model(inputs=[inputs], outputs=[conv8])
```

## Hyperparameters

- Batch size = 100
- Max Epochs = 30
- PATIENCE = 4
- Initial Learning rate = 1e-3


```python
EPOCHS = 30
BATCH_SIZE = batch_size

# Early stopping
PATIENCE = 4

# Learning rate
LEARNING_RATE = 1e-3
DECAY_FACTOR = 0.75
STEP_LR = 4
```

## Ottimizzatore e callbacks
Per evitare l'overfitting del modello utilizzo l'Early Stopping: dopo PATIENCE epoche in cui non si registra un miglioramento sulla loss, l'allenamento termina.

Per quanto riguarda la modifica del LearningRate durante il training, ho scelto di utilizzare lo Step Decay
Schedule, ovvero il learning rate viene moltiplicato con un fattore 0*75 ogni 4 epoche. In tal modo si verrà a creare una discesa del learning rate a scalino. In questo modo è possibile ridurre il tempo di training e migliorare le performance.



```python
EARLY_STOPPING = EarlyStopping(monitor='loss', mode='min', patience=PATIENCE)
```


```python
OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate = LEARNING_RATE)

def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)

LR_SCHEDULE = step_decay_schedule(initial_lr=LEARNING_RATE, decay_factor=DECAY_FACTOR, step_size=STEP_LR)

```

## Training


```python
model = nn()
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy', 'mean_squared_error'])
```

### Summary of the model


```python
model.summary()
keras.utils.plot_model(model, show_shapes=True, dpi=76)
```


```python
history = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=(20000 // BATCH_SIZE), callbacks = [EARLY_STOPPING, LR_SCHEDULE])
```

## Visualization of training history


```python
def plot_metrics(history):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))  # Imposta la dimensione della figura

    axes[0].plot(history.history['loss'], color='blue')
    axes[0].set_title("Binary CE")
    axes[0].set_xlabel("Epoca")
    axes[0].grid(True)

    axes[1].plot(history.history['accuracy'], color='red')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    axes[2].plot(history.history['lr'], color='green')
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    axes[3].plot(history.history['mean_squared_error'], color='green')
    axes[3].set_title("Mean squared error")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

# Utilizzo della funzione
plot_metrics(history)

```

# Visualization of generated images


```python
def showImages(x, y, z, customNcols=3, customFigSize=(12,8)):
  # Adjust figsize as needed
  fig, axes = plt.subplots(nrows=3, ncols=customNcols, figsize=customFigSize)

  # Plot image on each subplot
  for i in range(customNcols):
    axes[0,i].imshow(x[i])  # Use cmap='gray' if your images are grayscale
    axes[0,i].axis('off') # Turn off axis
    axes[0,i].title.set_text(f"img maschera {i}")
    axes[1,i].imshow(y[i])
    axes[1,i].axis('off')
    axes[1,i].title.set_text(f"img ricostruita {i}")
    axes[2,i].imshow(z[i])
    axes[2,i].axis('off')
    axes[2,i].title.set_text(f"img reale {i}")

  plt.tight_layout()
  plt.show()

test_x, test_y = next(test_generator)
inpainted_image = model.predict(test_x)
showImages(test_x, inpainted_image, test_y)
```

## Evaluation of the model
The mse is calculated on 10000 images generated from the test set for 10 times and the mean value and standard deviation are given.


```python
no_batch = 10000 // BATCH_SIZE
mse_scores = []

for i in range(10):
  for j in range(no_batch):
    mse = tf.keras.losses.MeanSquaredError()
    test_x, test_y = next(test_generator)
    prediction = model.predict(test_x, verbose=0)
    mse_value = mse(test_y, prediction)
    mse_scores.append(mse_value)

mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f'Mean MSE: {mean_mse}, Std MSE: {std_mse}')
```
