# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2 * np.random.randn(1000, 1)  # noise added
N = X.shape[0]

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


def cost(w):
    return .5 / Xbar.shape[0] * np.linalg.norm(y - Xbar.dot(w), 2)**2


n_point = 50


# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi * a).reshape(2, 1)


def batch_grad(w, i, rd_id):
    sum = np.zeros_like(w)
    for it in range(n_point):
        if i + it >= N:
            break
        sum = sum + sgrad(w, i + it, rd_id)
    return sum / (it + 1)


def GD_minibatch_NAG(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]

    w_last_check = w_init
    N = X.shape[0]

    iter_check_w = 1
    count = 0
    for it in range(10):
        # shuffle data
        rd_id = np.random.permutation(N)
        i = 0
        while i < N:
            count += 1
            g = batch_grad(w[-1] - gamma * v[-1], i, rd_id)

            v_new = gamma * v[-1] + eta * g
            w_new = w[-1] - v_new

            w.append(w_new)
            v.append(v_new)

            if count % iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / \
                        len(w_init) < 1e-3:
                    return (w, len(w))
                w_last_check = w_this_check
            i += n_point

    return (w, len(w))


w_init = np.array([[2], [1]])
(w1, it1) = GD_minibatch_NAG(w_init, sgrad, .3, .9)
print('Solution found by GD mini-batch: w = ',
      w1[-1].T, ',\nafter %d iterations.' % (it1 + 1))

if it1 > 200:
    scale = it1 // 50
else:
    scale = 1

fig, ax = plt.subplots()
fig.set_tight_layout(True)

x = np.linspace(0, 1, 2, endpoint=True)

ax.plot(X.T, y.T, 'b.')
ax.set_xlabel('')

line, = ax.plot(x, w_init[0] + w_init[1] * x, 'k')


def update(i):
    i = i * scale
    label = 'iter ' + str(i) + '/' + str(it1)
    ax.set_xlabel(label)

    ax.plot(x, w1[i][0] + w1[i][1] * x, 'r')


anim = FuncAnimation(fig, update, frames=np.arange(
    0, it1 // scale), interval=100)
anim.save('GD_minibatch_NAG_ex2.gif', dpi=80, writer='imagemagick')


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
    ii = ii * scale
    if ii == 0:
        ax.cla()
        CS = ax.contour(Xg, Yg, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        animlist = ax.clabel(
            CS, inline=.1, fontsize=10, manual=manual_locations)
        ax.plot(w_lr[0], w_lr[1], 'go')
    else:
        animlist = ax.plot([w1[ii - scale][0], w1[ii][0]],
                           [w1[ii - scale][1], w1[ii][1]], 'r-')
    animlist = ax.plot(w1[ii][0], w1[ii][1], 'ro', markersize = 4)
    xlabel = 'iter = %d/%d' % (ii, it1)
    # xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(sgrad(w1[ii]))
    ax.set_xlabel(xlabel)
    return animlist, ax


anim1 = FuncAnimation(
    fig, update_2, frames=np.arange(0, it1 // scale), interval=100)
anim1.save('GD_minibatch_NAG_ex2_contours.gif', dpi=100, writer='imagemagick')

loss = np.zeros((len(w1), 1))
for i in range(len(w1)):
    loss[i] = cost(w1[i])

plt.figure()
plt.cla()
if it1 < 50:
    plt.plot(range(it1), loss[:it1], 'b')
else:
    plt.plot(range(50), loss[:50], 'b')
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('LR with SGD, first 50 iters')
plt.savefig('GD_minibatch_NAG_ex2_loss.png')
