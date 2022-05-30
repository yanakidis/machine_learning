import oracles
import time
from scipy.special import expit
import numpy as np


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise AttributeError('Loss function is not binary logistic')
        else:
            self.loss_function = loss_function
        self.alpha = step_alpha
        self.beta = step_beta
        self.accuracy = tolerance
        self.max_iter = max_iter
        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
        else:
            self.l2_coef = 0

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None, flag=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        def acc(w):
            sk = X_val.dot(w)
            sk[sk >= 0] = 1
            sk[sk < 0] = -1
            return np.mean(sk == y_val)
        if trace is True:
            history = dict()
            time_list = []
            func_list = []
            accuracy = []
            start_time = time.time()
            k = 1
            w = w_0
            w_next = w - self.alpha / (k ** self.beta) * self.get_gradient(X, y, w)
            f = self.get_objective(X, y, w)
            f_next = self.get_objective(X, y, w_next)
            func_list.append(f)
            if flag is True:
                accuracy.append(acc(w))
            end_time = time.time()
            time_list.append(end_time - start_time)
            start_time = time.time()
            while (k - 1 != self.max_iter) & (abs(f_next - f) >= self.accuracy):
                k += 1
                learning_rate = self.alpha / (k ** self.beta)
                w = w_next
                f = f_next
                func_list.append(f)
                if flag is True:
                    accuracy.append(acc(w))
                w_next = w - learning_rate * self.get_gradient(X, y, w)
                f_next = self.get_objective(X, y, w_next)
                end_time = time.time()
                time_list.append(end_time - start_time)
                start_time = time.time()
            func_list.append(f_next)
            if flag is True:
                accuracy.append(acc(w_next))
            end_time = time.time()
            time_list.append(end_time - start_time)
            self.w = w_next
            if flag is True:
                history['accuracy'] = accuracy
            history['time'] = time_list
            history['func'] = func_list
            return history
        else:
            w = w_0
            k = 1
            w_next = w - self.alpha / (k ** self.beta) * self.get_gradient(X, y, w)
            while (k-1 != self.max_iter) & (abs(self.get_objective(X, y, w_next) -
                                            self.get_objective(X, y, w)) >= self.accuracy):
                k += 1
                learning_rate = self.alpha / (k ** self.beta)
                w = w_next
                w_next = w - learning_rate * self.get_gradient(X, y, w)
            self.w = w_next

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        tmp = X.dot(self.w)
        tmp[tmp >= 0] = 1
        tmp[tmp < 0] = -1
        return tmp

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        tmp = X.dot(self.w)
        tmp = expit(tmp)
        tmp2 = 1 - tmp
        X = np.array([tmp2, tmp]).T
        return X

    def get_objective(self, X, y, w):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        my_oracle = oracles.BinaryLogistic(l2_coef=self.l2_coef)
        return my_oracle.func(X, y, w)

    def get_gradient(self, X, y, w):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        my_oracle = oracles.BinaryLogistic(l2_coef=self.l2_coef)
        return my_oracle.grad(X, y, w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise AttributeError('Loss function is not binary logistic')
        else:
            self.loss_function = loss_function
        self.alpha = step_alpha
        self.beta = step_beta
        self.accuracy = tolerance
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_seed = random_seed
        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
        else:
            self.l2_coef = 0

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_val=None, y_val=None, flag=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        def BatchGenerator(list_of_sequences, batch_size, shuffle=False):
            length = len(list_of_sequences[0])
            amount = len(list_of_sequences)
            temp = list_of_sequences
            if shuffle is True:
                for j in range(0, len(temp)):
                    arr = np.array(temp[j])
                    np.random.shuffle(arr)
                    temp[j] = list(arr)
            i = 0
            if length % batch_size == 0:
                k = 0
            else:
                k = 1
            for k in range(0, length // batch_size + k):
                tmp = []
                for s in range(0, amount):
                    tmp.append(temp[s][i:i + batch_size])
                i += batch_size
                yield tmp

        def get_batch(bg):
            try:
                batch = bg.__next__()
            except StopIteration:
                bg = BatchGenerator(list_of_sequences=[[k for k in range(0, X.shape[0])]], batch_size=self.batch_size,
                                    shuffle=True)
                batch = bg.__next__()
            return batch[0]
        
        def acc(w):
            sk = X_val.dot(w)
            sk[sk >= 0] = 1
            sk[sk < 0] = -1
            return np.mean(sk == y_val)
        np.random.seed(self.random_seed)
        bg = BatchGenerator(list_of_sequences=[[k for k in range(0, X.shape[0])]], batch_size=self.batch_size,
                            shuffle=True)
        if trace is True:
            history = dict()
            history['epoch_num'] = []
            history['time'] = []
            history['func'] = []
            history['weights_diff'] = []
            history['accuracy'] = []
            start_time = time.time()
            w = w_0
            w_epoch = w
            k = 1
            p = get_batch(bg)
            w_next = w - self.alpha / (k ** self.beta) * self.get_gradient(X[p], y[p], w)
            epoch = k * self.batch_size / X.shape[0]
            if epoch > log_freq:
                end_time = time.time()
                history['epoch_num'].append(epoch)
                history['time'].append(end_time - start_time)
                history['func'].append(self.get_objective(X[p], y[p], w_next))
                history['weights_diff'].append(np.sum((w_next - w_epoch)*(w_next - w_epoch)))
                if flag is True:
                    history['accuracy'].append(acc(w_epoch))
                w_epoch = w_next
                start_time = time.time()
            f_next = self.get_objective(X[p], y[p], w_next)
            f = self.get_objective(X[p], y[p], w)
            while (k - 1 != self.max_iter) & (abs(f_next - f) >= self.accuracy):
                f = f_next
                k += 1
                learning_rate = self.alpha / (k ** self.beta)
                w = w_next
                p = get_batch(bg)
                w_next = w - learning_rate * self.get_gradient(X[p], y[p], w)
                epoch = k * self.batch_size / X.shape[0]
                if len(history['epoch_num']) == 0:
                    a = 0
                else:
                    a = history['epoch_num'][-1]
                if epoch - a > log_freq:
                    end_time = time.time()
                    history['epoch_num'].append(epoch)
                    history['time'].append(end_time - start_time)
                    history['func'].append(self.get_objective(X[p], y[p], w_next))
                    history['weights_diff'].append(np.sum((w_next - w_epoch) * (w_next - w_epoch)))
                    if flag is True:
                        history['accuracy'].append(acc(w_epoch))
                    w_epoch = w_next
                    start_time = time.time()
                f_next = self.get_objective(X[p], y[p], w_next)
            self.w = w_next
            return history
        else:
            w = w_0
            k = 1
            p = get_batch(bg)
            w_next = w - self.alpha / (k ** self.beta) * self.get_gradient(X[p], y[p], w)
            while (k - 1 != self.max_iter) & (abs(self.get_objective(X[p], y[p], w_next) -
                                              self.get_objective(X[p], y[p], w)) >= self.accuracy):
                k += 1
                learning_rate = self.alpha / (k ** self.beta)
                w = w_next
                p = get_batch(bg)
                w_next = w - learning_rate * self.get_gradient(X[p], y[p], w)
            self.w = w_next
