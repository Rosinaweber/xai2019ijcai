from numpy import asarray
from numpy import zeros
from numpy import amin, amax
from numpy import array, hstack
from numpy.random import randint
from numpy import sum as npsum
from numpy import argmax
from numpy import argsort
from numpy import bincount
from numpy import dot


class CBR(object):
    """Case-Based Reasoning Model"""

    def __init__(self, instances, types, weights=None):
        self._attr_init(instances)
        self._normalize()
        self._diff_func(types)
        if weights is None:
            self._learn_rlf()
        else:
            self._w = asarray([weights])

    def _attr_init(self, instances):
        # save cases
        self._X = instances[:, 1:]
        self._y = instances[:, 0].astype(int)

        # calculate shape
        self._m, self._n = self._X.shape

        # initialize weights
        self._w = zeros(shape=(1, self._n))

    def _normalize(self):
        # compute attributes min max
        X_min = amin(self._X, axis=0)
        X_max = amax(self._X, axis=0)

        # scale function convert attributes to [0, 1]
        self._scale = lambda q: (q - X_min) / (X_max - X_min)

        # scale attributes
        self._X = self._scale(self._X)

    def _diff_func(self, types):
        # function list
        func = []

        # assign funciont for each attributes based on type
        for i, t in enumerate(types):
            if t == 'd' or t == 'discrete':
                fn = lambda a, b: 1 if a == b else 0
            elif t == 'c' or t == 'continuous':
                fn = lambda a, b: 1 - abs(a - b)
            else:
                fn = lambda a, b: 0
            func.append(fn)

        # save function list
        self._f = tuple(func)

    def _calc_dist(self, X, q):
        dist = zeros(shape=X.shape)

        # calculate for each attribute
        for i, diff in enumerate(self._f):

            # compute difference
            arr = array([diff(q[i], x) for x in X[:, i]])

            # merge result to distance matrix
            dist[:, i] = arr

        return dist

    def _learn_rlf(self):
        # initialize weights
        self._w = zeros(shape=(1, self._n))

        # random select instance m times
        for i in randint(self._m, size=self._m):
            # randomly selected instance
            q = self._X[i, :]
            a = self._y[i]

            # remaining instances
            r = list(j for j in range(self._m) if i != j)
            X = self._X[r, :]
            y = self._y[r]

            # calculate distance and global similarity without weights
            d = self._calc_dist(X, q)
            s = npsum(d, axis=1)

            # update weight by nearest hit
            # (first part of formula from paper: diff(A, R, H) / m)
            self._w += d[y == a][argmax(s[y == a])] / self._m

            # update weight by sum of the nearest miss for each class
            for c in set(self._y):
                # C != class(R)
                if c == a:
                    continue

                # calculate P(C)
                p = sum(self._y == c) / self._m

                # (second part in sum formula: P(C) * diff(A, R, M(C)) / m)
                self._w -= p * d[y == c][argmax(s[y == c])] / self._m

    def _predict(self, X, q, k):
        # normalize q
        q = self._scale(q)

        # calculate distance
        d = self._calc_dist(X, q)

        # calculate weighted distance
        w = (d @ self._w.T).T

        # get the index of k-nearest neighbors
        p = list(argsort(-w)[0, 0:k])

        # return the mode (most common) answer
        return argmax(bincount(self._y[p]))

    def predict(self, q, k):
        return self._predict(self._X, q, k)

    def loocv(self, k):
        # init list to save validation result
        v = []

        # select instance m times
        for i in range(self._m):
            q = self._X[i, :]
            a = self._y[i]

            # remaining instances
            r = list(j for j in range(self._m) if i != j)
            X = self._X[r, :]
            y = self._y[r]

            # predict using remaining instances
            p = self._predict(X, q, k)

            # add result
            v.append(p == a)

        # compute accuracy
        return sum(v) / len(v)

    def loocvs(self, ks):
        # init dictionary to save validation results
        v = {k: [] for k in ks}

        # select instance m times
        for i in range(self._m):
            q = self._X[i, :]
            a = self._y[i]

            # remaining instances
            r = list(j for j in range(self._m) if i != j)
            X = self._X[r, :]
            y = self._y[r]

            # for each k
            for k in ks:

                # predict using remaining instances
                p = self._predict(X, q, k)

                # add result
                v[k].append(p == a)

        # compute accuracy
        return {k: sum(v[k]) / len(v[k]) for k in ks}


class CBR_GD(CBR):
    """Case-Based Reasoning Model with Gradient Descent"""

    def __init__(self, instances, types):
        self._attr_init(instances)
        self._normalize()
        self._diff_func(types)
        self._w = zeros(shape=(1, self._n))

    def _gd_epoch(self, lr):
        # predicted value
        p = []

        # select instance m times
        for i in range(self._m):
            q = self._X[i, :]
            a = self._y[i]

            # remaining instances
            r = list(j for j in range(self._m) if i != j)
            X = self._X[r, :]
            y = self._y[r]

            # predict with current weight
            p.append(self._predict(X, q, k=1))

        # calculate gradient
        g = dot(self._X.T, (self._y - asarray([p])).T) / self._X.shape[0]

        # update weights
        self._w += g.T * lr
