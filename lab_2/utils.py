import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    def identity_v(i):
        tmp = np.zeros_like(w)
        tmp[i] = 1
        return tmp

    result = []
    for i in range(0,w.shape[0]):
        val = (function(w + eps * identity_v(i)) - function(w))/eps
        result.append(val)
    return np.array(result)
