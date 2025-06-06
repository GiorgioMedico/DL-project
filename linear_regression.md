A simple animation showing linear regression


```python
#linear regression example
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
```


```python
from matplotlib import rc

# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')
```


```python
#a bunch of points on the plain
x = np.array([1,2.2,3,4,5,6,7,8,9,10])
y = np.array([14,12,13,15,11,9,8,4,2,1])

#gradient of the quadratic loss
def grad(a,b):
    d = y - (a*x + b)      #derivative of the loss
    da = - np.sum(d * x)   #derivative of d w.r.t. a
    db = - np.sum(d)       #derivative of d w.r.t. b
    return(da,db)
```


```python
lr = 0.001
epochs = 2000

#step 1
a = np.random.rand()
b = np.random.rand()
params=[a,b]

fig = plt.figure()
plt.plot(x,y,'ro')
line, = plt.plot([], [], lw=2)
```


```python
def init():
    #current approximation
    line.set_data([x[0],x[9]],[a*x[0]+b,a*x[9]+b])
    return line,

def step(i):
    a,b=params
    da,db = grad(a,b)
    if i%100==0:
      print("current loss = {}".format(np.sum((y-a*x-b)**2)))
    params[0] = a - lr*da
    params[1] = b - lr*db
    ##### for animation
    line.set_data([x[0],x[9]],[a*x[0]+b,a*x[9]+b])
    #time.sleep(.01)
    return line,

anim = animation.FuncAnimation(fig, step, init_func=init, frames=epochs, interval=1, blit=True, repeat=False)
```

The animation will be visualized at the end of the excution.


```python
anim
```
