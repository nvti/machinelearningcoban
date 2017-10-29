# generate data
# list of points
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X0[:, -1] = np.array([4, 2])


X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis = 1)
# Xbar
X = np.concatenate((np.ones((1, 2 * N)), X), axis = 0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    pocket = []
    for it in range(100):
        # mix data
        mix_id = np.random.permutation(N)

        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)
        num_miss = np.sum(y != h(w[-1], X))

        if it == 0:  # first step
            pocket.append((w[-1], num_miss))
        else:
            if num_miss < pocket[-1][1]:
                pocket.append((w[-1], num_miss))

        if num_miss == 0:
            break
        # if has_converged(X, y, w[-1]):
        #     break
    return (w, mis_points, pocket, it)


d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m, p, it1) = perceptron(X, y, w_init)
print(m, p, it1)


def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1 * x11 + w0) /
                                     w2, -(w1 * x12 + w0) / w2], 'k')
    else:
        x10 = -w0 / w1
        return plt.plot([x10, x10], [-100, 100], 'k')


it = len(w)
fig, ax = plt.subplots(figsize=(5, 5))
plt.cla()
# points
plt.plot(X0[0, :], X0[1, :], 'b^', markersize = 8, alpha = .8)
plt.plot(X1[0, :], X1[1, :], 'ro', markersize = 8, alpha = .8)
plt.axis([0, 6, -2, 4])
draw_line(p[-1][0])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.title('PLA pocket. Miss points: ' + str(p[-1][1]))
ax.set_xlabel('w = ' + str(p[-1][0]))

plt.savefig('PLA_PocketAlgorithm.png')
plt.show()
