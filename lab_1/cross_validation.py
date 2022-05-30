import numpy as np
import nearest_neighbors as near_n


def kfold(n, n_folds):
    indexes = np.arange(n)
    indexes = np.array_split(indexes, n_folds)
    list_of_indexes = list()
    for i in range(n_folds):
        tmp1 = indexes[i]
        if i != 0:
            tmp2 = indexes[0]
        else:
            tmp2 = indexes[1]
        for j in range(n_folds):
            if i != 0:
                if j != i and j != 0:
                    tmp2 = np.concatenate((tmp2, indexes[j]))
            else:
                if j != i and j != 1:
                    tmp2 = np.concatenate((tmp2, indexes[j]))
        list_of_indexes.append((tmp2, tmp1))
    return list_of_indexes


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    def predict_t(obj):
        classes = np.unique(obj.y)
        if w is False:
            l1 = np.arange(0)
            for c in classes:
                b = neighbors[:len(i[1])+1, :k]
                a = np.tile(obj.y, (len(i[1]), 1))
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
                b = neighbors[1][:len(i[1]) + 1, :k]
                a = np.tile(obj.y, (len(i[1]), 1))
                d = (np.arange(b.shape[0] * b.shape[1]) // b.shape[1], b.flatten())
                tmp = a[d].reshape((b.shape[0], b.shape[1]))
                a = np.sum(np.array([tmp == c]) *
                           1 / (neighbors[0][:len(i[1])+1, :k] + eps), axis=2)
                if l1.size > 1:
                    l1 = np.concatenate((l1, a), axis=0)
                else:
                    l1 = a
            return np.argmax(l1, axis=0)
    if cv is None:
        cv = kfold(X.shape[0], 3)
    res = dict()
    w = kwargs['weights']
    for k in k_list:
        res[k] = np.arange(0)
    NN = near_n.KNNClassifier(k=k_list[len(k_list) - 1], **kwargs)
    for i in cv:
        NN.fit(X[i[0]], y[i[0]])
        neighbors = NN.find_kneighbors(X[i[1]], w)
        for k in k_list:
            proportion = np.mean(predict_t(NN) == y[i[1]])
            res[k] = np.append(res[k], proportion)
    return res
