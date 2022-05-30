from sklearn.neighbors import NearestNeighbors
import numpy as np
import distances as dist


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=1):
        if (k < 1):
            raise AttributeError("k can't be less than 1, k = ", k)
        else:
            self.k = k
        if (strategy != 'my_own'):
            if (strategy == 'brute'):
                self.neigh = NearestNeighbors(n_neighbors=self.k, metric=metric,
                                              algorithm=strategy)
            else:
                self.neigh = NearestNeighbors(n_neighbors=self.k, metric='euclidean',
                                              algorithm=strategy)
        self.strategy = strategy
        self.metric = metric
        self.weights = bool(weights)
        if (test_block_size < 1):
            raise AttributeError("Test block size can't be less than 1, size = ",
                                 test_block_size)
        else:
            self.test_block_size = test_block_size

    def fit(self, X, y):
        if self.strategy != 'my_own':
            self.neigh.fit(X)
        else:
            self.X = X
        self.y = y

    def find_kneighbors(self, X, return_distance):
        if self.strategy != 'my_own':
            return self.neigh.kneighbors(X=X, return_distance=return_distance)
        else:
            if (self.metric == 'euclidean'):
                dist_1 = dist.euclidean_distance(X, self.X)
            else:
                dist_1 = dist.cosine_distance(X, self.X)
            dist_2 = np.argsort(dist_1, axis=1)
            if (self.k < dist_2.shape[1]):
                dist_2 = dist_2[:, :self.k]
            if return_distance is True:
                dist_1.sort(axis=1)
                if (self.k < dist_1.shape[1]):
                    dist_1 = dist_1[:, :self.k]
                return (dist_1, dist_2)
            else:
                return dist_2

    def predict(self, X):
        neighbors = self.find_kneighbors(X, self.weights)
        classes = np.unique(self.y)
        if self.weights is False:
            l1 = np.arange(0)
            for c in classes:
                b = neighbors[:X.shape[0] + 1]
                a = np.tile(self.y, (X.shape[0], 1))
                d = (np.arange(b.shape[0] * b.shape[1]) // b.shape[1], b.flatten())
                tmp = a[d].reshape((b.shape[0], b.shape[1]))
                a = np.sum([tmp == c], axis=2)
                if l1.size > 1:
                    l1 = np.concatenate((l1, a), axis=0)
                else:
                    l1 = a
            return np.argmax(l1, axis=0)
        else:
            eps = 1e-5
            l1 = np.arange(0)
            for c in classes:
                b = neighbors[1][:X.shape[0] + 1]
                a = np.tile(self.y, (X.shape[0], 1))
                d = (np.arange(b.shape[0] * b.shape[1]) // b.shape[1], b.flatten())
                tmp = a[d].reshape((b.shape[0], b.shape[1]))
                a = np.sum(np.array([tmp == c]) *
                           1 / (neighbors[0][:X.shape[0] + 1] + eps), axis=2)
                if l1.size > 1:
                    l1 = np.concatenate((l1, a), axis=0)
                else:
                    l1 = a
            return np.argmax(l1, axis=0)
