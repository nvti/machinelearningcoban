import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print 'Number of classes    : %d' % len(np.unique(iris_y))
print 'Number of data points: %d' % len(iris_y)


X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=50)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))

print("Norm 2, weights = \'uniform\'")
for i in range(1, 10):
    clf = neighbors.KNeighborsClassifier(n_neighbors = i, p = 2,
                                         weights = 'uniform')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy of " + str(i) + "NN with major voting: %.2f %%" %
          (100 * accuracy_score(y_test, y_pred)))

print("Norm 2, weights = \'distance\'")
for i in range(1, 10):
    clf = neighbors.KNeighborsClassifier(n_neighbors = i, p = 2,
                                         weights = 'distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy of " + str(i) + "NN with major voting: %.2f %%" %
          (100 * accuracy_score(y_test, y_pred)))


def myweight(distances):
    sigma2 = .5  # we can change this number
    return np.exp(-distances**2 / sigma2)


print("Norm 2, weights = custom")
for i in range(1, 10):
    clf = neighbors.KNeighborsClassifier(n_neighbors = i, p = 2,
                                         weights = myweight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy of " + str(i) + "NN with major voting: %.2f %%" %
          (100 * accuracy_score(y_test, y_pred)))
