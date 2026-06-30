import numpy as np


class V1:

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
        The only major difference is that in this version, the function used to calculate the Learning Rate is a simple if else statement, meant to emulate the table, (check readme):
        ```
        grad_norm = np.linalg.norm(self.g_hat)
        abs_kappa = np.abs(self.kappa_hat)

        kappa_high_threshold = 1.0
        grad_high_threshold = 1.0

        kappa_high = abs_kappa > kappa_high_threshold
        grad_high = grad_norm > grad_high_threshold
        kappa_positive = self.kappa_hat > 0

        if (kappa_positive and kappa_high) or (not kappa_positive and kappa_high and grad_high):
            eta = 0.001
        else:
            eta = 0.1

        return float(np.clip(eta, self.min_lr, self.max_lr))
        ```
        ============================================================

        V1 version is the simplest version, and hence performs the worst (when compared to the other versions)  on all types of terrain
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
        return (self.fn.gradients(self.x + epsilon * self.g) - self.g) / epsilon

    def calculate_curvature(self):
        denominator = self.g.T @ self.g
        if denominator < 1e-12:
            return 0.0
        return (self.g.T @ self.calculate_hessian_change()) / denominator

    def track_curvature_and_gradient(self):
        self.t += 1
        self.g_bar = self.alpha * self.g_bar + (1 - self.alpha) * self.g
        kappa = self.calculate_curvature()
        self.kappa_bar = self.beta * self.kappa_bar + (1 - self.beta) * kappa
        self.g_hat = self.g_bar / (1 - self.alpha ** self.t)
        self.kappa_hat = self.kappa_bar / (1 - self.beta ** self.t)
        return self.kappa_hat, self.g_hat

    def get_learning_rate(self):
        grad_norm = np.linalg.norm(self.g_hat)
        abs_kappa = np.abs(self.kappa_hat)

        kappa_high_threshold = 1.0
        grad_high_threshold = 1.0

        kappa_high = abs_kappa > kappa_high_threshold
        grad_high = grad_norm > grad_high_threshold
        kappa_positive = self.kappa_hat > 0

        if (kappa_positive and kappa_high) or (not kappa_positive and kappa_high and grad_high):
            eta = 0.001
        else:
            eta = 0.1

        return float(np.clip(eta, self.min_lr, self.max_lr))

    def step(self):
        self.g = self.calculate_gradients()
        kappa_hat, g_hat = self.track_curvature_and_gradient()
        eta = self.get_learning_rate()
        self.x = self.x - eta * g_hat

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