import autograd.numpy as np
from autograd import grad


class Sphere:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        return np.sum(x ** 2)

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.zeros(self.n)


class IllConditionedQuadratic:
    def __init__(self, n, condition_number=1e6):
        self.n = n
        self.lambdas = np.logspace(0, np.log10(condition_number), n)
        self._grad = grad(self.function)

    def function(self, x):
        return np.sum(self.lambdas * x ** 2)

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.zeros(self.n)


class RosenbrockND:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.ones(self.n)


class RastriginND:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        return 10 * self.n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.zeros(self.n)


class AckleyND:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / self.n)

        return term1 + term2 + 20 + np.e

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.zeros(self.n)


class GriewankND:
    def __init__(self, n):
        self.n = n
        self.indices = np.arange(1, n + 1)
        self._grad = grad(self.function)

    def function(self, x):
        term1 = np.sum(x ** 2) / 4000
        term2 = np.prod(np.cos(x / np.sqrt(self.indices)))
        return term1 - term2 + 1

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.zeros(self.n)


class SchwefelND:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        return 418.9829 * self.n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.full(self.n, 420.968746)


class LevyND:
    def __init__(self, n):
        self.n = n
        self._grad = grad(self.function)

    def function(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2* (1 + np.sin(2 * np.pi * w[-1] ) ** 2)
        return term1 + term2 + term3

    def gradients(self, x):
        return self._grad(x)

    def minima(self):
        return np.ones(self.n)


class StyblinskiTangND:

    _X_STAR = -2.903534

    def __init__(self, n):
        self.n = n

    def function(self, x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

    def gradients(self, x):
        return 2 * x**3 - 16 * x + 2.5

    def minima(self):
        return self._X_STAR * np.ones(self.n)


class DoubleWellND:

    def __init__(self, n):
        self.n = n

    def function(self, x):
        return np.sum(x ** 4 - 4 * x ** 2)

    def gradients(self, x):
        return 4 * x ** 3 - 8 * x

    def minima(self):
        return np.sqrt(2) * np.ones(self.n)

