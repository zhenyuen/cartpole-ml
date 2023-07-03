from CartPole_ import CartPole, remap_angle
import numpy as np
from numpy.random import default_rng
from scipy.stats import qmc, linregress
from scipy.optimize import minimize
from numba import njit

from shared import(
    POS_LOW,
    POS_HIGH,
    VEL_LOW,
    VEL_HIGH,
    ANG_VEL_LOW,
    ANG_VEL_HIGH,
    ANG_LOW,
    ANG_HIGH,
    FORCE_LOW,
    FORCE_HIGH,
    STABLE_POS,
    STABLE_VEL,
    STABLE_ANG,
    STABLE_ANG_VEL,
    UNSTABLE_POS,
    UNSTABLE_VEL,
    UNSTABLE_ANG,
    UNSTABLE_ANG_VEL,
    SEED,
)

class LinearObserver(CartPole):
    def __init__(self, visual=False, N=11, dim=4):
        print(dim)
        super().__init__(visual)
        self.N = N
        self.n_data = 2 ** self.N
        self.w = np.zeros((dim, dim))
        self.mse = 0
        self.dim = dim

    @classmethod
    def initialize_model(cls, N, with_action):
        dim = 5 if with_action else 4
        return cls(False, N=N, dim=dim)
        
    def save_model(self, fname):
        with open(fname, 'wb') as f:
            np.save(f, self.N)
            np.save(f, self.n_data)
            np.save(f, self.w)
            np.save(f, self.mse)
            np.save(f, self.dim)
    
    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.N = np.load(f)
            self.n_data = np.load(f)
            self.w = np.load(f)
            self.sigma = np.load(f)
            self.mse = np.load(f)
            self.dim = np.load(f)
        
    def fit(self, target, enable_remap=True):
        def simulate(x):
            if self.dim == 4: x = np.hstack([x, 0])
            _, state_hist = target.simulate(state=x, remap=enable_remap, n_steps=1)
            return state_hist[1, :] - x[:4]
        
        def remap(x):
            x[2] = remap_angle(x[2])
            return x
        
        sampler = qmc.Sobol(
            d=self.dim,
            seed=SEED,
        )

        X = sampler.random_base2(self.N)
        X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
        X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        
        if self.dim == 5: X[:, 4] = X[:, 4] * (FORCE_HIGH - FORCE_LOW) + FORCE_LOW
        print(X.shape)
        
        y = np.apply_along_axis(
            simulate,
            axis=1,
            arr=X,
        )

        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        change = X @ w

        # if enable_remap: change = np.apply_along_axis(remap, axis=1, arr=change)

        diff_train = y - change
        mse = np.sqrt(np.sum(diff_train**2)) / y.size
        print(w.shape)
        self.w = w
        self.mse = mse

        return y, change

    def performAction(self, action=0.0):
        if self.dim == 4:
            x0 = np.array(
                    [
                        self.cart_location,
                        self.cart_velocity,
                        self.pole_angle,
                        self.pole_velocity,
                    ]
                )
        else:
            x0 = np.array(
                    [
                        self.cart_location,
                        self.cart_velocity,
                        self.pole_angle,
                        self.pole_velocity,
                        action
                    ]
                )
            
        x = numba_perform_action(x0, self.w)
        
        (
            self.cart_location,
            self.cart_velocity,
            self.pole_angle,
            self.pole_velocity,
        ) = x

    def setParams(self, w, mse_train=-1, mse_test=-1):
        self.w = w
        self.mse_test = mse_train
        self.mse_train = mse_test

    def setState(self, state):
        return super().setState(state)

    def resetParams(self):
        self.w = np.zeros(self.n_basis)
        self.mse = 0


@njit 
def numba_perform_action(x, w):
    change = x @ w
    x = x[:4] + change[0]
    return x