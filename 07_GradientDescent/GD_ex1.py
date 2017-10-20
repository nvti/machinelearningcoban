# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def grad(x):
    return 2 * x + 5 * np.cos(x)


def cost(x):
    return x**2 + 5 * np.sin(x)


def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)


(x1, it1) = myGD1(.1, -5)

print('Solution x1 = %f, cost = %f, obtained after %d iterations' %
      (x1[-1], cost(x1[-1]), it1))

fig, ax = plt.subplots()
fig.set_tight_layout(True)

X = np.arange(-5, 5, 0.001)

ax.plot(X, cost(X), 'b')

cur_point = ax.plot([-5], cost(-5), 'ko')
new_point = ax.plot([-5], cost(-5), 'ro')


def update(i):
    label = 'timestep {0}'.format(i)
    ax.set_xlabel(label)


anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=500)
plt.show()
