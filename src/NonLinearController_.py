import numpy as np
from scipy.optimize import minimize
from itertools import count
from scipy.stats import qmc
from numba import njit


POS_LOW = -10
POS_HIGH = 10
VEL_LOW = -10
VEL_HIGH = 10
ANG_VEL_LOW = -15
ANG_VEL_HIGH = 15
ANG_LOW = -np.pi
ANG_HIGH = np.pi

SEED = 10

class NonLinearController():
    def __init__(self, w, xi, omega) -> None:
        self.w = w
        self.xi = xi
        self.omega = omega
        self.prev = 0

    def get_params(self):
        return self.xi, self.w, self.omega

    def get_params_as_np_array(self):
        size = self.xi.size
        xi = self.xi.flatten()

        w = np.pad(self.w, (0, size - self.w.size), mode='constant')
        omega = np.pad(self.omega, (0, size - self.omega.size), mode='constant')
    
        return np.vstack((xi, w, omega))
    
    @classmethod
    def format_sol_as_np_array(cls, xi, w, omega):
        size = xi.size
        xi = xi.flatten()

        w = np.pad(w, (0, size - w.size), mode='constant')
        omega = np.pad(omega, (0, size - omega.size), mode='constant')
    
        return np.vstack((xi, w, omega))
    
    @classmethod
    def format_sol_from_np_array(cls, sol):
        p = np.reshape(sol.x, (-1, sol.x.size // 3))
        xi, w, omega = p
        xi = np.reshape(xi, (-1 ,4))
        w = w[:4]
        omega = omega[:xi.shape[0]]
        return xi, w, omega

    @classmethod
    def get_non_linear_controller(cls, m=4):
        # Sample basis locations with sobol
        sampler = qmc.Sobol(
            d=4,
            seed=SEED,
        )
        xi = sampler.random_base2(m)

        cart_pos_lim = (-2, 2)
        cart_vel_lim = (-2.5, 2.5)
        pole_vel_lim = (-5, 5)
        pole_ang_lim = (-np.pi, np.pi)
        force_lim = (-6, 6)

        xi[:, 0] = xi[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
        xi[:, 1] = xi[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
        xi[:, 2] = xi[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
        xi[:, 3] = xi[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW

        # xi[:, 0] = xi[:, 0] * 1
        # xi[:, 1] = xi[:, 1] * 1
        # xi[:, 2] = np.ones_like(xi[:, 2]) * np.pi * 0.75
        # xi[:, 3] = xi[:, 3] * 1

        # xi[:, 0] = xi[:, 0] * 4 + (-2)
        # xi[:, 1] = xi[:, 1] * 2 + (-0.5)
        # xi[:, 2] = xi[:, 2] * 0.4 + (-0.2)
        # xi[:, 3] = xi[:, 3] * (2) + (-1)

        # w = np.ones(4) * 0.5
        # w[2] = 13

        w = np.array([ 1.82882224,  2.20082917, 19.37292826,  3.02057818]) / 4
        omega = np.ones(xi.shape[0]) * 6
        return cls(w, xi, omega)
    
    def fit_limit_iter(self, model, x0, p0, n_steps, verbose=False):        
        callback = NonLinearController._optimize_linear_policy_callback if verbose else None
        func = lambda p: NonLinearController.total_loss(p, model, x0, n_steps)
        model.reset()
        options = {"maxiter": 50}
        return minimize(func, p0, method='Nelder-Mead', callback=callback, options=options)
    
    def fit(self, model, x0, p0, n_steps, verbose=False):        
        callback = NonLinearController._optimize_linear_policy_callback if verbose else None
        func = lambda p: NonLinearController.total_loss(p, model, x0, n_steps)
        model.reset()
        return minimize(func, p0, method='Nelder-Mead', callback=callback)
    
    def fit_nelder_mead(self, model, x0, p0, n_steps, verbose=False):
        return self.fit(model, x0, p0, n_steps, verbose=False)

    def fit_bfgs(self, model, x0, p0, n_steps, verbose=False):
        callback = NonLinearController._optimize_linear_policy_callback if verbose else None
        func = lambda p: NonLinearController.total_loss(p, model, x0, n_steps)
        model.reset()
        options = {"eps": 4, "disp": verbose}
        return minimize(func, p0, callback=callback, method='BFGS', options=options)


    @classmethod
    def total_loss(cls, p, model, x0, n_steps):
        p = np.reshape(p, (-1, p.size // 3))
        xi, w, omega = p

        xi = np.reshape(xi, (-1 ,4))
        w = w[:4]
        omega = omega[:xi.shape[0]]

        model.setState(x0)
        total_loss = model.loss()
        action = NonLinearController._eval(x0, xi, w, omega)
        # print("Initial action force:", action)
        for _ in range(n_steps):
            model.performAction(action)
            state = model.getState()
            # print(state)
            total_loss += model.loss()
            action = NonLinearController._eval(state, xi, w, omega)
        model.reset()
        return total_loss

    def feedback_action(self, x):
        grad = x - self.prev
        tol = 0.02
        if grad[2] >= 0:
            if np.abs(grad[2]) >= (np.pi + tol): sign = -1
            else: sign = 1
        else:
            if np.abs(grad[2]) >= (np.pi + tol): sign = 1
            else: sign = -1

        self.prev = x
        k_m = numba_k(x[:4], self.xi, self.w)
        return np.dot(k_m, self.omega) * sign
    

    @classmethod
    def _optimize_linear_policy_callback(cls, p, counter=count()):
        i = next(counter)
        print(f"Iter {i: >3}: {p}")

    @classmethod
    def _eval(cls, x, xi, w, omega):
        k_m = numba_k(x[:4], xi, w)
        return np.dot(k_m, omega)
    
    @classmethod
    def _K(
        cls,
        x,
        xi,
        w,
    ):
        d = x.size
        xi = np.reshape(xi, (-1, d))
        expo = np.zeros(xi.shape[0])
        W = np.outer(w, w)

        def se(x, xi):
            r = np.reshape((x - xi), (d, -1))

            return (r.T @ W @ r) / 2

        for j in range(xi.shape[0]):
            expo[j] = se(
                x,
                xi[j, :],
            )
        
        return np.exp(-expo)
    

    def fit_only_omega(self, model, x0, p0, n_steps, verbose=False):
        callback = NonLinearController._optimize_linear_policy_callback if verbose else None

        w = self.w

        func = lambda p: NonLinearController.total_loss_only_omega(p, self.xi, w, model, x0, n_steps)
        model.reset()
        return minimize(func, p0, callback=callback, method='Nelder-Mead')
    
    @classmethod
    def total_loss_only_omega(cls, p, xi, w, model, x0, n_steps):
        model.setState(x0)
        total_loss = model.loss()
        action = NonLinearController._eval(x0, xi, w[0], p)
        # print("Initial action force:", action)
        for i, _ in enumerate(range(1, n_steps + 1)):
            print("action force:", action)
            model.performAction(action)
            state = model.getState()
            # print(state)
            total_loss += model.loss()
            action = NonLinearController._eval(state, xi, w[i], p)
        model.reset()
        return total_loss


@njit
def numba_k(x, xi, sigma):
    xi = np.reshape(xi, (-1, 4))
    N = x.shape[0]
    M = xi.shape[0]
    expo = np.zeros(M)

    def se(x, xi):
        r = (x - xi) / sigma
        return (np.dot(r, r)) / 2

    
    for j in range(xi.shape[0]):
        expo[j] = se(
            x,
            xi[j, :],
        )
   
    return np.exp(-expo)



class OscillatingController():
    def __init__(self) -> None:
        self.prev = 0
        self.count = 0
        self.weight = 0.5

    def feedback_action(self, state):
        grad = state - self.prev
        tol = 0.02
        if grad[2] >= 0:
            if np.abs(grad[2]) >= (np.pi + tol): sign = -1
            else: sign = 1
        else:
            if np.abs(grad[2]) >= (np.pi + tol): sign = 1
            else: sign = -1

        self.prev = state
        # return 1 * np.exp(-state[2]) * sign
        return 2 * sign
    
        # t = state / self.weight
        # coeff = 1 - np.exp(-np.dot(t, t) / 2)
        # grad = state[2] - self.prev
        # self.prev = state[2]
        # return grad * coeff * 10


class OscillatingController2():
    # To incoporate position feedback
    def __init__(self) -> None:
        self.prev = np.zeros(4)
        self.count = 0
        self.weight = 0.5

    def feedback_action(self, state):
        grad = state - self.prev
        tol = 0.02
        if grad[2] >= 0:
            if np.abs(grad[2]) >= (np.pi + tol): sign = -1
            else: sign = 1
        else:
            if np.abs(grad[2]) >= (np.pi + tol): sign = 1
            else: sign = -1


        # if state[0] > 3: sign = 1
        # elif state[0] < 3: sign = -1
        # return 1 * np.exp(-state[2]) * sign
        # self.prev = state

        return np.abs(state[2]) * 10 / np.pi * sign
        # return np.abs(state[2]) * 10 / np.pi * np.sign(state[3])
        # t = state / self.weight
        # coeff = 1 - np.exp(-np.dot(t, t) / 2)
        # grad = state[2] - self.prev
        # self.prev = state[2]
        # return grad * coeff * 10


class EnsembleController():
    def __init__(self, controllers) -> None:
        self.controllers = controllers
        self.state = 0

    def feedback_action(self, state):
        if self.state == 0:
            # Gain momentum phase, 0.34395644,  1.24783025,  1.43247604,  2.94037778
            if np.abs(state[1]) > 1 and np.abs(state[3]) > 1 and np.abs(state[2]) < np.pi / 2:
                self.state = 1
                print(state)
            else:
                return self.controllers[0].feedback_action(state)

        if self.state == 1:
            # Decay phase
            # 0.00438198, -0.05325854, -0.03511533, -0.00613946
            # if np.all(state == np.array([0.00438198, -0.05325854, -0.03511533, -0.00613946])):
            if np.all(np.abs(state) < 0.1):
                self.state = 2
                print(state)
            else:
                return self.controllers[1].feedback_action(state)
        
        if self.state == 2:
            # Linear controller to stabilize
            return self.controllers[2].feedback_action(state)
        return 0.0
