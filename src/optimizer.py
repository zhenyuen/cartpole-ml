import numpy as np
from scipy.optimize import minimize
from itertools import count
from CartPole_ import loss

class Optimizer():
    def __init__(self) -> None:
        self.sol_based_on_initial = {}
        self.min_loss_based_on_initial = {}
        self.min_loss = float('inf')
        self.sol = None
    
    def optimize_linear(self, model, controller, initial_state, initial_guess, verbose, loss_func, time=0.0, remap=True):
        if loss_func == 1:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss
        elif loss_func == 2:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss_with_feedback
        p, loss = self._optimize_linear(model, controller, initial_state, initial_guess, callback, func, time, remap)
        return p, loss
    
    def optimize_non_linear(self, model, controller, initial_state, initial_guess, verbose, loss_func, time=0.0, remap=True, target=0):
        if loss_func == 1:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss_non_linear_omega_only
        if loss_func == 2:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss_non_linear_omega_only_2
        if loss_func == 3:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss_non_linear_omega_only_3
        if loss_func == 4:
            callback = Optimizer._optimize_linear_policy_callback if verbose else None
            func = self.total_loss_non_linear_omega_only_4
        p, loss = self._optimize_non_linear(model, controller, initial_state, initial_guess, callback, func, time, remap, target)
        return p, loss
    
    def _optimize_linear(self, model, controller, s0, p0, callback, func, time, remap):
        inner_func = lambda p: func(model, controller, s0, p, time, remap)
        options = None
        sol = minimize(inner_func, p0, method='Nelder-Mead', callback=callback, options=options)
        
        p = sol.x
        loss = func(model, controller, s0, p, time, remap, training=False)

        if loss < self.min_loss:
            self.sol = p
            self.min_loss = loss
        key = tuple(s0)
        if key in self.min_loss_based_on_initial:
            if self.min_loss_based_on_initial[key] > loss:
                self.sol_based_on_initial[key] = p
                self.min_loss_based_on_initial[key] = loss
        else:
            self.sol_based_on_initial[key] = p
            self.min_loss_based_on_initial[key] = loss
        
        return p, loss
    

    def _optimize_non_linear(self, model, controller, s0, p0, callback, func, time, remap, target):
        inner_func = lambda p: func(model, controller, s0, p, time, remap, target)
        options = None
        sol = minimize(inner_func, p0, method='Nelder-Mead', callback=callback, options=options)
        
        p = sol.x
        loss = func(model, controller, s0, p, time, remap, target, training=False)

        if loss < self.min_loss:
            self.sol = p
            self.min_loss = loss
        key = tuple(s0)
        if key in self.min_loss_based_on_initial:
            if self.min_loss_based_on_initial[key] > loss:
                self.sol_based_on_initial[key] = p
                self.min_loss_based_on_initial[key] = loss
        else:
            self.sol_based_on_initial[key] = p
            self.min_loss_based_on_initial[key] = loss
        
        return p, loss
    
    def total_loss(self, model, controller, s0, p, time, remap, training=False):
        temp = controller.p[:]
        controller.p = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        for s in state_hist:
            total_loss += loss(s)
        controller.p = temp
        return total_loss
    
    def total_loss_with_feedback(self, model, controller, s0, p, time, remap, training=False):
        # take into consideration feedback force too.
        temp = controller.p[:]
        controller.p = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        force_hist = np.zeros((state_hist.shape[0], 1))

        for i, s in enumerate(state_hist):
            force_hist[i] = controller.feedback_action(s)

        for s, f  in zip(state_hist, force_hist):
            total_loss += loss(np.hstack([s, f]))

        controller.p = temp # Restore 
        return total_loss
    
    def total_loss_non_linear_omega_only(self, model, controller, s0, p, time, remap, target, training=False):
        # take into consideration feedback force too + logarithm
        temp = controller.omega[:]
        controller.omega = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        force_hist = np.zeros((state_hist.shape[0], 1))

        for i, s in enumerate(state_hist):
            force_hist[i] = controller.feedback_action(s)

        for s, f in zip(state_hist, force_hist):
            total_loss += np.log(loss(np.hstack([s, f])))

        controller.p = temp # Restore 
        return total_loss
    
    def total_loss_non_linear_omega_only_2(self, model, controller, s0, p, time, remap, target, training=False):
        # Define custom loss function,
        # Escape if any omega larger than 20
        if np.any(np.abs(p)) > 20: return 1

        temp = controller.omega[:]
        controller.omega = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        force_hist = np.zeros((state_hist.shape[0], 1))

        for i, s in enumerate(state_hist):
            force_hist[i] = controller.feedback_action(s)

        for s, f in zip(state_hist, force_hist):
            total_loss += np.log(loss(np.hstack([s, f])))

        controller.p = temp # Restore 
        return total_loss
    
    def total_loss_non_linear_omega_only_3(self, model, controller, s0, p, time, remap, target, training=False):
        # Define custom loss function,
        # Escape if any omega larger than 20
        # Ignore original objecive, your goal now is to make the pole angle tend towards 0
        if np.any(np.abs(p)) > 20: return 1

        temp = controller.omega[:]
        controller.omega = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        force_hist = np.zeros((state_hist.shape[0], 1))

        for i, s in enumerate(state_hist):
            force_hist[i] = controller.feedback_action(s)

        for s, f in zip(state_hist, force_hist):
            total_loss += np.log(loss(np.hstack([s, f])))

        controller.p = temp # Restore 
        return total_loss
    
    def total_loss_non_linear_omega_only_4(self, model, controller, s0, p, time, remap, target, training=False):
        # Define custom loss function,
        # Escape if any omega larger than 20
        # Ignore original objecive, your goal now is to make the pole angle tend towards 0
        temp = controller.omega[:]
        controller.omega = p
        _, state_hist = model.simulate_with_feedback(state=s0, remap=remap, controller=controller, time=time)
        total_loss = 0
        force_hist = np.zeros((state_hist.shape[0], 1))

        for i, s in enumerate(state_hist):
            force_hist[i] = controller.feedback_action(s)

        for s, f in zip(state_hist, force_hist):
            total_loss += np.log(loss(s, target))

        controller.p = temp # Restore 
        return total_loss
    
    def get_optimal_param_for_state(self, s0):
        key = tuple(s0)
        if key in self.min_loss_based_on_initial:
            p = self.sol_based_on_initial[key]
            loss = self.min_loss_based_on_initial[key]
            return p, loss
        return None, None

    @classmethod
    def _optimize_linear_policy_callback(cls, p, counter=count()):
        i = next(counter)
        print(f"Iter {i: >3}: {p}")

def get_next_search_space_limits(prev_opt, factor=10):
    delta = np.abs(prev_opt) / factor
    limits = []

    for i, p in enumerate(prev_opt):
        limits.append([p - delta[i], p + delta[i]])
    
    return limits


def get_minimum_loss(loss_hist):
    ind = np.unravel_index(np.argmin(loss_hist, axis=None), loss_hist.shape)
    return ind, loss_hist[ind]
