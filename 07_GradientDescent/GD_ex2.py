# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2 * np.random.randn(1000, 1)  # noise added

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ', w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1 * x0


def grad(w):
    N = Xbar.shape[0]
    XbarW = Xbar.dot(w)
    XbarWY = XbarW - y
    return 1 / N * Xbar.T.dot(XbarWY)


def cost(w):
    N = Xbar.shape[0]
    return .5 / N * np.linalg.norm(y - Xbar.dot(w), 2)**2


def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
    return g


def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False


print('Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))


def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)


w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print('Solution found by GD: w = ',
      w1[-1].T, ',\nafter %d iterations.' % (it1 + 1))


fig, ax = plt.subplots()
fig.set_tight_layout(True)

x = np.linspace(0, 1, 2, endpoint=True)

ax.plot(X.T, y.T, 'b.')
ax.set_xlabel('')

line, = ax.plot(x, w_init[0] + w_init[1] * x, 'k')


def update(i):
    label = 'iter ' + str(i) + '/' + str(it1) + ', grad = ' + str(grad(w1[i]))
    ax.set_xlabel(label)

    ax.plot(x, w1[i][0] + w1[i][1] * x, 'r')


anim = FuncAnimation(fig, update, frames=np.arange(0, it1), interval=200)
anim.save('GD_ex2.gif', dpi=80, writer='imagemagick')


N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2 / N
b1 = 2 * np.sum(X) / N
c1 = np.linalg.norm(X, 2)**2 / N
d1 = -2 * np.sum(y) / N
e1 = -2 * X.T.dot(y) / N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(1.5, 7.0, delta)
yg = np.arange(0.5, 4.5, delta)
Xg, Yg = np.meshgrid(xg, yg)

Z = a1 + Xg**2 + b1 * Xg * Yg + c1 * Yg**2 + d1 * Xg + e1 * Yg

plt.figure()
fig, ax = plt.subplots()
fig.set_tight_layout(True)
plt.cla()


def update_2(ii):
    if ii == 0:
        plt.cla()
        CS = plt.contour(Xg, Yg, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        animlist = plt.clabel(
            CS, inline=.1, fontsize=10, manual=manual_locations)
        plt.plot(w_lr[0], w_lr[1], 'go')
    else:
        animlist = plt.plot([w1[ii - 1][0], w1[ii][0]],
                            [w1[ii - 1][1], w1[ii][1]], 'r-')
    animlist = plt.plot(w1[ii][0], w1[ii][1], 'ro', markersize = 4)
    xlabel = 'iter = %d/%d' % (ii, it1)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w1[ii]))
    ax.set_xlabel(xlabel)
    return animlist, ax


anim1 = FuncAnimation(fig, update_2, frames=np.arange(0, it1), interval=200)
anim1.save('GD_ex2_contours.gif', dpi=100, writer='imagemagick')
