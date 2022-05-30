import numpy as np
from scipy.special import expit
from scipy import sparse


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        margin = (- X.dot(w.T) * y.T).T
        matr = np.exp(margin.astype(np.float32))
        matr = np.log1p(matr)
        regul = self.l2_coef/2 * np.sum(w*w)
        Q = 1/X.shape[0] * np.sum(matr) + regul
        return Q

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
        def prod(X, y):
            if sparse.issparse(X):
                return X.multiply(y.reshape(-1, 1))
            else:
                return (X * y.reshape(-1, 1))

        grad_Q = - 1 / X.shape[0] * np.sum(prod(prod(X, y), expit(- X.dot(w)
                                                                  * y.astype(np.float32))), axis=0) + self.l2_coef * w
        return np.ravel(grad_Q)
