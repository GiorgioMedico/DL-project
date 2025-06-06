## Fooling neural networks

Here we show how to fool a neural network using a gradient ascent technique over the input.


```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras import losses 
import numpy as np
import matplotlib.pyplot as plt
```

Let us start importing the VGG16 model.


```
model = VGG16(weights='imagenet', include_top=True)
#model.summary()
```

Now, we load an image (in our case, an elephant)


```
from google.colab import files
uploaded = files.upload()
```

Next, we classify it. 

VGG16 is higly confident it is an elephant.


```
img = image.load_img('elephant2.jpg', target_size=(224, 224))

x0 = image.img_to_array(img)
x = np.expand_dims(x0, axis=0)
preds = model.predict(x)
print("label = {}".format(np.argmax(preds)))
print('Predicted:', decode_predictions(preds, top=3)[0])

xd = image.array_to_img(x[0])
imageplot = plt.imshow(xd)
plt.show()
```

Now we try to convert the image into something different: a tiger shark, with label 3.


```
output_index = 3 #tiger shark

expected_output = np.zeros(1000)
expected_output[output_index] = 1
expected_output = K.variable(np.reshape(expected_output,(1,1000)))

```

Now we simply iterate the gradient ascent technique for a sufficent number of steps, working on a copy of the original image


```
input_img_data = np.copy(x)

# run gradient ascent for 50 steps
for i in range(50):
    print("iteration n. {}".format(i))
    with tf.GradientTape() as g:
      x = K.variable(input_img_data)
      y = model(x)
      loss = tf.keras.losses.categorical_crossentropy(y,expected_output)
    res = y[0]
    print("elephant prediction: {}".format(res[386]))
    print("tiger shark prediction: {}".format(res[3]))
    grads_value = g.gradient(loss, x)[0]
    print(grads_value.shape)
    ming = np.min(grads_value)
    maxg = np.max(grads_value)
    #print("min grad = {}".format(ming))
    #print("max grad = {}".format(maxg))
    scale = 1/(maxg -ming)
    #brings gradients to a sensible value
    input_img_data -= grads_value * scale

```

At the end, VGG16 is extremely confident he is looking at a tiger shark


```
preds = model.predict(input_img_data)
print("label = {}".format(np.argmax(preds)))
print('Predicted:', decode_predictions(preds, top=3)[0])
```

Let us look at the resulting image (we both print the original and the processed image)


```
nimg = input_img_data[0]
nimg = image.array_to_img(img)

plt.figure(figsize=(10,5))
ax = plt.subplot(1, 2, 1)
plt.title("elephant")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.imshow(xd)
ax = plt.subplot(1, 2, 2)
plt.imshow(nimg)
plt.title("tiger shark")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

imageplot = plt.imshow(img)
plt.show()
```

We just fooled the neural network! 
