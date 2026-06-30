from autograd import grad
import autograd.numpy as np


# 1. Adam:
class Adam:
    def __init__(self, function_to_optimize, beta, alpha, x, epsilon, scheduler):
        self.fn = function_to_optimize
        self.beta = beta
        self.alpha = alpha
        self.t = 0
        self.m = 0
        self.v = 0
        self.x = x
        self.epsilon = epsilon

        self.scheduler = scheduler

        self.history_m_corrected = []
        self.history_v_corrected = []
        self.history_g = []
        self.history_adam_term = []
        self.history_x = []
        self.history_euclidean_distance_minima_from_convergence = []

    def calculate_grads(self):
        return self.fn.gradients(self.x)

    def track_momentum_and_velocity(self):
        self.t += 1
        g = self.calculate_grads()
        self.m = self.beta * self.m + (1 - self.beta) * g
        self.v = self.alpha * self.v + (1 - self.alpha) * g ** 2
        corrected_m = self.m / (1 - self.beta ** self.t)
        corrected_v = self.v / (1 - self.alpha ** self.t)
        return corrected_m, corrected_v, g

    def step(self):
        self.lr = self.scheduler()
        correct_m, correct_v, g = self.track_momentum_and_velocity()
        adam_term = self.lr * (correct_m / (np.sqrt(correct_v) + self.epsilon))
        self.x = self.x - adam_term
        self.scheduler.step()

        self.history_m_corrected.append(correct_m)
        self.history_v_corrected.append(correct_v)
        self.history_g.append(g)
        self.history_adam_term.append(adam_term)
        self.history_x.append(self.x)
        self.history_euclidean_distance_minima_from_convergence.append(
            np.abs(np.linalg.norm(self.x - self.fn.minima())))

        return self.x, correct_m, correct_v, g

    def check_minima_versus_coordinates(self, tolerance):
        grad = self.calculate_grads()
        return np.linalg.norm(grad) < tolerance

    def track_history(self):
        return {'m_history': self.history_m_corrected,
                'v_history': self.history_v_corrected,
                'g_history': self.history_g,
                'adam_term_history': self.history_adam_term,
                'x_history': self.history_x,
                'convergence_history': self.history_euclidean_distance_minima_from_convergence}


# 2. NAdam:
class NAdam:
    def __init__(self, function_to_optimize, beta, alpha, x, epsilon, scheduler):
        self.fn = function_to_optimize
        self.beta = beta
        self.alpha = alpha
        self.t = 0
        self.m = 0
        self.v = 0
        self.x = x
        self.scheduler = scheduler
        self.epsilon = epsilon

        self.history_m_corrected = []
        self.history_v_corrected = []
        self.history_g = []
        self.history_nadam_term = []
        self.history_x = []
        self.history_euclidean_distance_minima_from_convergence = []

    def calculate_grads(self):
        return self.fn.gradients(self.x)

    def track_momentum_and_velocity(self):
        self.t += 1
        g = self.calculate_grads()
        self.m = self.beta * self.m + (1 - self.beta) * g
        self.v = self.alpha * self.v + (1 - self.alpha) * g ** 2
        corrected_m = self.m / (1 - self.beta ** self.t)
        corrected_v = self.v / (1 - self.alpha ** self.t)
        m_nesterov = self.beta * corrected_m + (1 - self.beta) * g / (1 - self.beta ** self.t)
        return m_nesterov, corrected_v, g

    def step(self):
        self.lr = self.scheduler()
        correct_m, correct_v, g = self.track_momentum_and_velocity()
        adam_term = self.lr * (correct_m / (np.sqrt(correct_v) + self.epsilon))
        self.x = self.x - adam_term
        self.scheduler.step()

        self.history_m_corrected.append(correct_m)
        self.history_v_corrected.append(correct_v)
        self.history_g.append(g)
        self.history_adam_term.append(adam_term)
        self.history_x.append(self.x)
        self.history_euclidean_distance_minima_from_convergence.append(
            np.abs(np.linalg.norm(self.x - self.fn.minima())))

        return self.x, correct_m, correct_v, g

    def check_minima_versus_coordinates(self, tolerance):
        grad = self.calculate_grads()
        return np.linalg.norm(grad) < tolerance

    def track_history(self):
        return {'m_history': self.history_m_corrected,
                'v_history': self.history_v_corrected,
                'g_history': self.history_g,
                'nadam_term_history': self.history_nadam_term,
                'x_history': self.history_x,
                'convergence_history': self.history_euclidean_distance_minima_from_convergence}


# 3. AdamW:
class AdamW:
    def __init__(self, function_to_optimize, beta, alpha, x, scheduler, epsilon, regularization_factor):
        self.fn = function_to_optimize
        self.beta = beta
        self.alpha = alpha
        self.t = 0
        self.m = 0
        self.v = 0
        self.x = x
        self.scheduler = scheduler
        self.epsilon = epsilon
        self._lambda = regularization_factor

        self.history_m_corrected = []
        self.history_v_corrected = []
        self.history_g = []
        self.history_adam_term = []
        self.history_x = []
        self.history_euclidean_distance_minima_from_convergence = []

    def calculate_grads(self):
        return self.fn.gradients(self.x)

    def track_momentum_and_velocity(self):
        self.t += 1
        g = self.calculate_grads()
        self.m = self.beta * self.m + (1 - self.beta) * g
        self.v = self.alpha * self.v + (1 - self.alpha) * g ** 2
        corrected_m = self.m / (1 - self.beta ** self.t)
        corrected_v = self.v / (1 - self.alpha ** self.t)
        return corrected_m, corrected_v, g

    def step(self):
        self.lr = self.scheduler()
        correct_m, correct_v, g = self.track_momentum_and_velocity()
        adam_term = self.lr * (correct_m / (np.sqrt(correct_v) + self.epsilon))
        self.x = self.x - adam_term - self.lr * self._lambda * self.x
        self.scheduler.step()

        self.history_m_corrected.append(correct_m)
        self.history_v_corrected.append(correct_v)
        self.history_g.append(g)
        self.history_adam_term.append(adam_term)
        self.history_x.append(self.x)
        self.history_euclidean_distance_minima_from_convergence.append(
            np.abs(np.linalg.norm(self.x - self.fn.minima())))

        return self.x, correct_m, correct_v, g

    def check_minima_versus_coordinates(self, tolerance):
        grad = self.calculate_grads()
        return np.linalg.norm(grad) < tolerance

    def track_history(self):
        return {'m_history': self.history_m_corrected,
                'v_history': self.history_v_corrected,
                'g_history': self.history_g,
                'adam_term_history': self.history_adam_term,
                'x_history': self.history_x,
                'convergence_history': self.history_euclidean_distance_minima_from_convergence}


# 4. NAdamW:
class NAdamW:
    def __init__(self, function_to_optimize, beta, alpha, x, scheduler, epsilon, regularization_factor):
        self.fn = function_to_optimize
        self.beta = beta
        self.alpha = alpha
        self.t = 0
        self.m = 0
        self.v = 0
        self.x = x
        self.scheduler = scheduler
        self.epsilon = epsilon
        self._lambda = regularization_factor

        self.history_m_corrected = []
        self.history_v_corrected = []
        self.history_g = []
        self.history_nadam_term = []
        self.history_x = []
        self.history_euclidean_distance_minima_from_convergence = []

    def calculate_grads(self):
        return self.fn.gradients(self.x)

    def track_momentum_and_velocity(self):
        self.t += 1
        g = self.calculate_grads()
        self.m = self.beta * self.m + (1 - self.beta) * g
        self.v = self.alpha * self.v + (1 - self.alpha) * g ** 2
        corrected_m = self.m / (1 - self.beta ** self.t)
        corrected_v = self.v / (1 - self.alpha ** self.t)
        m_nesterov = self.beta * corrected_m + (1 - self.beta) * g / (1 - self.beta ** self.t)
        return m_nesterov, corrected_v, g

    def step(self):
        self.lr = self.scheduler()
        correct_m, correct_v, g = self.track_momentum_and_velocity()
        adam_term = self.lr * (correct_m / (np.sqrt(correct_v) + self.epsilon))
        self.x = self.x - adam_term - self.lr * self._lambda * self.x
        self.scheduler.step()

        self.history_m_corrected.append(correct_m)
        self.history_v_corrected.append(correct_v)
        self.history_g.append(g)
        self.history_x.append(self.x)
        self.history_euclidean_distance_minima_from_convergence.append(
            np.abs(np.linalg.norm(self.x - self.fn.minima())))

        return self.x, correct_m, correct_v, g

    def check_minima_versus_coordinates(self, tolerance):
        grad = self.calculate_grads()
        return np.linalg.norm(grad) < tolerance

    def track_history(self):
        return {'m_history': self.history_m_corrected,
                'v_history': self.history_v_corrected,
                'g_history': self.history_g,
                'nadam_term_history': self.history_nadam_term,
                'x_history': self.history_x,
                'convergence_history': self.history_euclidean_distance_minima_from_convergence}


# 5. Muon:
class Muon:
    def __init__(self, function_to_optimize, starting_point, scheduler, iterations_to_estimate_inverse_squareroot,
                 beta):
        self.fn = function_to_optimize
        self.x = starting_point
        self.iterations = iterations_to_estimate_inverse_squareroot
        self.scheduler = scheduler
        self.beta = beta
        self.g_bar = np.zeros_like(self.x)
        self.t = 0

        self.history_grad = []
        self.history_decomposed_grad = []
        self.history_euclidean_distance_minima_convergence = []
        self.history_x = [self.x.copy()]

    def calculate_gradients(self, x):
        return self.fn.gradients(x)

    def track_grads(self, x):
        grad = self.calculate_gradients(x)
        self.t += 1
        self.g_bar = self.beta * self.g_bar + (1 - self.beta) * grad
        g_hat = self.g_bar / (1 - self.beta ** self.t)
        return g_hat

    def decompose_gradient(self, grads):
        G = grads
        self.is_matrix = self.x.ndim == 2

        if self.is_matrix:
            A = G.T @ G
            A = A / (np.linalg.norm(A) + 1e-8)
            X = np.eye(A.shape[0])
            for _ in range(self.iterations):
                X = 0.5 * X @ (3 * np.eye(A.shape[0]) - X @ A @ X)
            QV_T = G @ X
        else:
            QV_T = G / np.linalg.norm(G)
        return QV_T

    def step(self):
        self.lr = self.scheduler()
        raw_grad = self.calculate_gradients(self.x)
        smoothened_grad = self.track_grads(self.x)
        decomposed_grad = self.decompose_gradient(smoothened_grad)
        self.x = self.x - self.lr * decomposed_grad
        self.scheduler.step()
        self.history_grad.append(raw_grad)
        self.history_decomposed_grad.append(decomposed_grad)
        self.history_euclidean_distance_minima_convergence.append(np.linalg.norm(self.x - self.fn.minima()))
        self.history_x.append(self.x)
        return self.x

    def track_history(self):
        return {
            "x_history": self.history_x,
            "convergence_history": self.history_euclidean_distance_minima_convergence,
            "decomposed_grads_history": self.history_decomposed_grad,
            "grads_history": self.history_grad
        }


# 6. BFGS
class BFGS:
    def __init__(self, function_to_optimize, starting_point):
        self.fn = function_to_optimize
        self.x = starting_point
        self.B = np.eye(self.x.size)
        self.a = 1e-4  # Armijo constant
        self.rho = 0.5  # shrinkage factor
        self.max_backtracks = 50
        self.initial_lr = 1.0

        self.history_x = []
        self.history_grad = []
        self.history_B = []
        self.history_convergence = []

    def calculate_gradients(self, x):
        return self.fn.gradients(x)

    def line_search(self, x, G, p):
        eta = self.initial_lr
        f_x = self.fn.function(x)
        G_dot_p = np.dot(G, p)
        for _ in range(self.max_backtracks):
            x_new = x + eta * p
            if self.fn.function(x_new) <= f_x + self.a * eta * G_dot_p:
                return eta
            eta *= self.rho
        return eta

    def step(self):
        G = self.calculate_gradients(self.x)
        p = -self.B @ G

        # Line search for step size
        eta = self.line_search(self.x, G, p)
        x_new = self.x + eta * p
        G_new = self.calculate_gradients(x_new)

        s = x_new - self.x
        y = G_new - G

        ys = np.dot(y, s)
        if ys > 1e-12:
            I = np.eye(self.B.shape[0])
            self.B = (I - np.outer(s, y) / ys) @ self.B @ (I - np.outer(y, s) / ys) + np.outer(s, s) / ys

        self.x = x_new

        self.history_x.append(self.x)
        self.history_grad.append(G)
        self.history_B.append(self.B)
        self.history_convergence.append(np.linalg.norm(self.x - self.fn.minima()))
        return self.x

    def track_history(self):
        return {
            "x_history": self.history_x,
            "gradient_history": self.history_grad,
            "B_history": self.history_B,
            "convergence_history": self.history_convergence
        }

# 7. AdaHessian
rng = np.random.default_rng()


class AdaHessian:
    def __init__(self, function_to_optimize, starting_point, beta_for_momentum, beta_for_velocity, scheduler):
        self.fn = function_to_optimize
        self.scheduler = scheduler
        self.x = starting_point
        self.g = grad(self.fn.function)
        self.m = 0
        self.v = 0
        self.beta1 = beta_for_momentum
        self.beta2 = beta_for_velocity
        self.eps = 1e-8
        self.t = 0

        self.history_x = []
        self.history_m = []
        self.history_v = []
        self.history_convergence = []

    def gradients(self):
        return self.g(self.x)

    def HVP(self, z):
        return grad(lambda x: np.dot(self.g(x), z))(self.x)

    def hutchinson(self):
        z = rng.choice([-1.0, 1.0], size=self.x.shape)
        hz = self.HVP(z)
        return z * hz

    def step(self):
        self.lr = self.scheduler()
        g = self.gradients()
        h_diag = self.hutchinson()
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (h_diag ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        self.history_x.append(self.x)
        self.history_m.append(self.m)
        self.history_v.append(self.v)
        self.history_convergence.append(np.linalg.norm(self.x - self.fn.minima()))

        self.x -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.scheduler.step()
        return self.x

    def track_history(self):
        return {
            "x_history": self.history_x,
            "m_history": self.history_m,
            "v_history": self.history_v,
            "convergence_history": self.history_convergence
        }
