import numpy as np
from scipy.optimize import minimize
from itertools import count
from scipy.stats import qmc
from sanitycheck import get_next_search_space
import matplotlib.pyplot as plt
from numba import njit
from CartPole_ import remap_angle


POS_LOW = -10
POS_HIGH = 10
VEL_LOW = -10
VEL_HIGH = 10
ANG_VEL_LOW = -15
ANG_VEL_HIGH = 15
ANG_LOW = -np.pi
ANG_HIGH = np.pi

SEED = 10

class LinearController():
    def __init__(self, p) -> None:
        self.p = p
        self.local_minima_losses = {}
        self.global_minima_loss = float('inf')


    def get_params(self):
        return self.p
    
    def get_global_minima_loss(self):
        return self.global_minima_loss
    
    def get_local_minima_losses(self):
        return self.local_minima_losses

    def print_local_minima_losses(self):
        for k, v in self.local_minima_losses.items():
            print(f"Minima: {k}, loss: {v}")
            

    @classmethod
    def get_linear_controller(cls, p0=[1.0, 1.0, 1.0, 1.0]):
        return cls(np.array(p0))


    @classmethod
    def total_loss(cls, p, model, x0, n_steps, min_loss, training=True):
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss


    def fit_controller(self, model, s0, p0, n_steps, verbose=False, loss_func=total_loss):        
        callback = LinearController._optimize_linear_policy_callback if verbose else None
        func = lambda p: loss_func(p, model, x0, n_steps, min_loss=self.global_minima_loss)
        model.reset()
        # options={"maxiter": 10}
        options = None
        sol = minimize(func, p0, method='Nelder-Mead', callback=callback, options=options)
        
        loss = loss_func(sol.x, model, x0, n_steps, min_loss=self.global_minima_loss, training=False)

        if loss < self.global_minima_loss:
            self.p = sol.x
            self.global_minima_loss = loss
        
        self.local_minima_losses[tuple(sol.x)] = loss
        return sol.x
    

    @classmethod
    def total_loss_2(cls, p, model, x0, n_steps, min_loss, training=True):
        # Key differences - constraint on cart position and applied force
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                # We constraint the cart displacement to 1 as we dont want the cart displacement to affect the response due to translation invariance 
                if np.any( np.abs(state) > np.array([1, 20, 20, 20]) ) or np.abs(action) > 5: # Further constraint search space
                    return n_steps
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss
    
    @classmethod
    def total_loss_3(cls, p, model, x0, n_steps, min_loss, training=True):
        # Key differences - Relax constraint force
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                # We constraint the cart displacement to 1 as we dont want the cart displacement to affect the response due to translation invariance 
                if np.any( np.abs(state) > np.array([5, 20, 20, 20]) ) or np.abs(action) > 20: # Further constraint search space
                    return n_steps
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss
    

    @classmethod
    def total_loss_4(cls, p, model, x0, n_steps, min_loss, training=True):
        t = np.array([10, 10, 10, 10])
        # Key differences - constraint on cart position and applied force
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                # Pole velocity keeps exploring to 40, we need to stop earlier
                if np.any( np.abs(state) > np.array([t[0], t[1], t[2], t[3]]) ) or np.abs(action) > 20: # Further constraint search space
                    prop = np.array([0.2, 0.2, 0.4, 0.2])
                    temp = prop * state / t
                    return n_steps * np.sum(np.abs(temp))
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss
    

    @classmethod
    def total_loss_5(cls, p, model, x0, n_steps, min_loss, training=True):
        t = np.array([10, 10, 10, 10])
        # Key differences - constraint on cart position and applied force
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                # Pole velocity keeps exploring to 40, we need to stop earlier
                if np.any( np.abs(state) > np.array([t[0], t[1], t[2], t[3]]) ) or np.abs(action) > 20: # Further constraint search space
                    prop = np.array([0.1, 0.1, 0.4, 0.4])
                    temp = prop * state / t
                    return n_steps * np.sum(np.abs(temp))
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss
    
    @classmethod
    def total_loss_6(cls, p, model, x0, n_steps, min_loss, training=True):
        t = np.array([10, 10, 10, 10])
        # Key differences - constraint on cart position and applied force
        model.setState(x0)
        total_loss = model.loss()
        action = LinearController._eval(x0, p)
        # print("Initial action force:", action)
        if training:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                if total_loss >= min_loss:
                    return min_loss
                # Pole velocity keeps exploring to 40, we need to stop earlier
                if np.any( np.abs(state) > np.array([t[0], t[1], t[2], t[3]]) ) or np.abs(action) > 20: # Further constraint search space
                    prop = np.array([0.3, 0.3, 0.8, 0.3])
                    temp = prop * state / t
                    return n_steps * np.sum(np.abs(temp))
                action = LinearController._eval(state, p)
        else:
            for _ in range(n_steps):
                model.performAction(action)
                state = model.getState()
                total_loss += model.loss()
                action = LinearController._eval(state, p)
        
        model.reset()
        return total_loss


    @classmethod
    def _optimize_linear_policy_callback(cls, p, counter=count()):
        i = next(counter)
        print(f"Iter {i: >3}: {p}")

    @classmethod # Keep this for previous versions
    def _eval(cls, x, p):
        return np.dot(x, p)
    
    def feedback_action(self, x):
        return np.dot(x, self.p)
    

    def fit_global_minima(self, model, x0, n_steps, n=10, limits=(-20, 20), loss_func=total_loss):
        if len(limits) == 2:
            limits = (limits, limits, limits, limits)

        p_range0 = np.linspace(limits[0][0], limits[0][1], n)
        p_range1 = np.linspace(limits[1][0], limits[1][1], n)
        p_range2 = np.linspace(limits[2][0], limits[2][1], n)
        p_range3 = np.linspace(limits[3][0], limits[3][1], n)
        
        for a, p0 in enumerate(p_range0):
            for b, p1 in enumerate(p_range1):
                for c, p2 in enumerate(p_range2):
                    for d, p3 in enumerate(p_range3):
                        p = np.array([p0, p1, p2, p3])
                        self.fit(model, x0, p, n_steps, verbose=False, loss_func=loss_func)
    
    def reset(self, reset_optimum=False):
        self.local_minima_losses = {}
        if reset_optimum:
            self.p = [1.0, 1.0, 1.0, 1.0]
            self.global_minima_loss = float('inf')



    def fit_global_minima_multi_level(self, model, x0, n_steps, n=10, max_epochs=5, factor=10.0, limits=(-20, 20), tolerance=1E-1, loss_func=total_loss):
        min_loss_prev = self.get_global_minima_loss()
        p_opt_prev =  self.get_params()

        try:
            iter(factor)
        except TypeError:
            factor = [factor] * max_epochs

        for i in range(max_epochs):
            self.reset()
            self.fit_global_minima(model, x0, n_steps, n, limits=limits, loss_func=loss_func)

            p_opt = self.get_params()
            min_loss = self.get_global_minima_loss()

            if np.all(p_opt == p_opt_prev): break

            print("Optimal params:", p_opt)
            print("Optimal loss:", self.get_global_minima_loss())

            if np.abs(min_loss - min_loss_prev) < tolerance: break

            min_loss_prev = min_loss
            p_opt_prev = p_opt
            limits = get_next_search_space(p_opt, factor[i])

        print("----- RESULTS -----")
        print(f"No. epochs: {i + 1}")
        print(f"Optimal params: {p_opt}")
        print(f"Optimal loss: {self.get_global_minima_loss()}")


    def fit(self, model, x0, p0, n_steps, verbose=False, loss_func=total_loss):        
        callback = LinearController._optimize_linear_policy_callback if verbose else None
        func = lambda p: loss_func(p, model, x0, n_steps, min_loss=self.global_minima_loss)
        model.reset()
        # options={"maxiter": 10}
        options = None
        sol = minimize(func, p0, method='Nelder-Mead', callback=callback, options=options)
        
        loss = loss_func(sol.x, model, x0, n_steps, min_loss=self.global_minima_loss, training=False)

        if loss < self.global_minima_loss:
            self.p = sol.x
            self.global_minima_loss = loss
        
        self.local_minima_losses[tuple(sol.x)] = loss
        return sol.x
    

    def simulate_rollout(self, model, x0, n_steps=100, p=None, enable_remap=True):
        if p is None: p = self.p

        model.setState(x0)
        x = x0[None, ...]
        action_hist = []
        time = np.arange(n_steps + 1) * 0.2

        action = LinearController._eval(x0, p)
        action_hist.append(action)

        for _ in range(n_steps):
            model.performAction(action, enable_remap)
            state = model.getState()
            if enable_remap: state[2] = remap_angle(state[2])

            x = np.vstack([x, state])
        
            action = LinearController._eval(state, p)
            action_hist.append(action)
        
        action_hist = np.array(action_hist)
        model.reset()

        l = 5
        colors = plt.cm.jet(np.linspace(0, 1, l))

        return time, x, action_hist, colors
    

