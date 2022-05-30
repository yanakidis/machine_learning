import numpy as np


def euclidean_distance(X, Y):
    X_norma_sq = np.sum(X*X, axis=1)
    Y_norma_sq = np.sum(Y*Y, axis=1)
    sk_prod = X.dot(Y.T)
    res = np.sqrt(X_norma_sq.T[:, np.newaxis] + Y_norma_sq[np.newaxis, :] - 2*sk_prod)
    return res


def cosine_distance(X, Y):
    sk_prod = X.dot(Y.T)
    X_norma = np.sqrt(np.sum(X * X, axis=1))
    Y_norma = np.sqrt(np.sum(Y * Y, axis=1))
    res = 1 - sk_prod / (X_norma.T[:, np.newaxis] * Y_norma[np.newaxis, :])
    return res
