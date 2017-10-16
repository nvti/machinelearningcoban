import numpy as np


def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]


X = np.array([[1, 2], [4, 5]])
k = 1
kmeans_init_centers(X, k)
