import numpy as np


class V2:

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
        The only major difference is that in this version, the function used to calculate the Learning Rate is a version of the risk function defined in the OptimizersMath file, meant to emulate the table, (check readme):
        ```
        g_distance = np.linalg.norm(self.g_hat)
        term_1 = 1 / (self.relu(self.kappa_hat) + 1 / g_distance)
        term_2 = (1 - self.sigmoid(self.revrelu(self.kappa_hat * g_distance)))
        lr = (term_1 + term_2)
        return float(np.clip(lr, self.min_lr, self.max_lr))
        ```

        where revrelu and relu are defined in the OptimizerMath.ipynb file under the main branch.
        ============================================================

        V2 version is a slight boost from V1
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

        self.history_x = []
        self.history_g_hat = []
        self.history_kappa_hat = []
        self.history_lr = []
        self.history_euclidean_distance_minima_convergence = []

    def calculate_gradients(self):
        return self.fn.gradients(self.x)

    def calculate_hessian_change(self, epsilon=1e-4):
        raw = (self.fn.gradients(self.x + epsilon * self.g) - self.g) / epsilon
        return raw/np.linalg.norm(self.g)

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
        return self.kappa_hat, self.g_hat

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
        kappa_hat, g_hat = self.track_curvature_and_gradient()
        eta = self.get_learning_rate()
        self.x = self.x - eta * g_hat / kappa_hat

        self.history_x.append(self.x.copy())
        self.history_g_hat.append(g_hat.copy())
        self.history_kappa_hat.append(kappa_hat)
        self.history_lr.append(eta)

        minima = self.fn.minima()
        if minima is not None:
            self.history_euclidean_distance_minima_convergence.append(
                np.linalg.norm(self.x - minima)
            )

        return self.x, eta, kappa_hat, g_hat

    def track_history(self):
        return {
            "x_history": self.history_x,
            "g_hat_history": self.history_g_hat,
            "kappa_hat_history": self.history_kappa_hat,
            "lr_history": self.history_lr,
            "convergence_history": self.history_euclidean_distance_minima_convergence
        }