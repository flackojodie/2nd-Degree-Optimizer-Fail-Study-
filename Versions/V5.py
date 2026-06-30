import numpy as np


class V5:

    def __init__(
            self,
            function_to_optimize,
            starting_point,
            minimum_learning_rate,
            maximum_learning_rate,
            alpha,
            beta
    ):
        """
        :param function_to_optimize: Terrain over we optimize
        :param starting_point: N dimensional array of an arbitrary start position on the terrain
        :param minimum_learning_rate: as the name suggests (1e-10)
        :param maximum_learning_rate: as the name suggests (1)
        :param alpha: EMA term for gradients (0.99)
        :param beta: EMA term for beta (0.999)

        ============================================================

        The only major difference is that in this version, the function used to calculate the Learning Rate is the LR risk function and we have integrated Adam
        behaviour into the optimizer: The LR function is the same as V2, but the update is different:
        ``` self.x = self.x - eta * self.m_hat / (np.sqrt(self.v_hat) + self.kappa_hat ** 0.025) ```

        ============================================================

        V5 version merges Adam and the curvature optimizer, but still fails on -ve curvature terrains and ML problem terrains.
        """
        self.fn = function_to_optimize
        self.min_lr = minimum_learning_rate
        self.max_lr = maximum_learning_rate
        self.alpha = alpha
        self.beta = beta
        self.t = 0
        self.x = np.array(starting_point, dtype=float)
        self.g = np.zeros_like(self.x)
        self.g_bar = np.zeros_like(self.x)
        self.g_hat = np.zeros_like(self.x)
        self.kappa_bar = 0.0
        self.kappa_hat = 0.0

        self.m = 0.0
        self.m_hat = 0.0
        self.v = 0.0
        self.v_hat = 0.0

        self.history_x = []
        self.history_g_hat = []
        self.history_kappa_hat = []
        self.history_lr = []
        self.history_euclidean_distance_minima_convergence = []

    def calculate_gradients(self):
        return self.fn.gradients(self.x)

    def calculate_hessian_change(self, epsilon=1e-4):
        raw = (self.fn.gradients(self.x + epsilon * self.g/np.linalg.norm(self.g)) - self.g) / epsilon
        return raw

    def calculate_curvature(self):
        g_norm = self.g/np.linalg.norm(self.g)
        denominator = g_norm.T @ g_norm
        return (g_norm.T @ self.calculate_hessian_change()) / denominator

    def track_curvature_and_gradient(self):
        self.t += 1
        self.g_bar = self.alpha * self.g_bar + (1 - self.alpha) * self.g
        kappa = self.calculate_curvature()
        self.kappa_bar = self.beta * self.kappa_bar + (1 - self.beta) * kappa ** 2
        self.g_hat = self.g_bar / (1 - self.alpha ** self.t)
        self.kappa_hat = self.kappa_bar / (1 - self.beta ** self.t)

        self.m = self.beta * self.m + (1 - self.beta) * self.g
        self.v = self.alpha * self.v + (1 - self.alpha) * self.g ** 2

        self.m_hat = self.m / (1 - self.beta ** self.t)
        self.v_hat = self.v / (1 - self.alpha ** self.t)

    def relu(self, x):
        return np.where(x > 0, x, 0)
    def revrelu(self, x):
        return np.where(x < 0, -x, 0)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_learning_rate(self):
        g_distance = np.linalg.norm(self.g_hat)
        term_1 = 1 / (self.relu(self.kappa_hat) + 1 / g_distance)
        term_2 = (1 - self.sigmoid(self.revrelu(self.kappa_hat * g_distance)))
        lr = (term_1 + term_2)
        return float(np.clip(lr, self.min_lr, self.max_lr))

    def step(self):
        self.g = self.calculate_gradients()
        self.track_curvature_and_gradient()
        eta = self.get_learning_rate()
        self.x = self.x - eta * self.m_hat / (np.sqrt(self.v_hat) + self.kappa_hat ** 0.025)

        self.history_x.append(self.x.copy())
        self.history_g_hat.append(self.g_hat.copy())
        self.history_kappa_hat.append(self.kappa_hat)
        self.history_lr.append(eta)

        minima = self.fn.minima()
        if minima is not None:
            self.history_euclidean_distance_minima_convergence.append(
                np.linalg.norm(self.x - minima)
            )

        return self.x, eta, self.kappa_hat, self.g_hat

    def track_history(self):
        return {
            "x_history": self.history_x,
            "g_hat_history": self.history_g_hat,
            "kappa_hat_history": self.history_kappa_hat,
            "lr_history": self.history_lr,
            "convergence_history": self.history_euclidean_distance_minima_convergence
        }