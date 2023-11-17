from CartPole_ import CartPole, remap_angle
import numpy as np
from numpy.random import default_rng
from scipy.stats import qmc, linregress
from scipy.optimize import minimize
from numba import njit
from scipy.signal import wiener


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


class NonLinearObserverFiltered(CartPole):
    def __init__(self, visual=False, N=11, M=5, dim=4):
        super().__init__(visual)
        self.N = N
        self.n_basis = (2 ** M) * 10
        self.n_data = 2 ** self.N
        self.w = np.zeros(self.n_basis)
        self.xi = np.zeros((self.n_basis, 5))
        self.sigma = np.zeros(5)
        self.mse = 0
        self.dim = dim
        eye = np.eye(4)
        filter_mat = wiener(eye, 2)
        self.filter = filter_mat

    @classmethod
    def initialize_model(cls, N, M, with_action):
        dim = 5 if with_action else 4
        return cls(False, N, M, dim)
        
    def save_model(self, fname):
        with open(fname, 'wb') as f:
            np.save(f, self.N)
            np.save(f, self.n_basis)
            np.save(f, self.n_data)
            np.save(f, self.w)
            np.save(f, self.xi)
            np.save(f, self.sigma)
            np.save(f, self.mse)
            np.save(f, self.dim)
    
    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.N = np.load(f)
            self.n_basis = np.load(f)
            self.n_data = np.load(f)
            self.w = np.load(f)
            self.xi = np.load(f)
            self.sigma = np.load(f)
            self.mse = np.load(f)
            self.dim = np.load(f)

    def fit(self, target, enable_remap=True, sigma=None):
        def simulate(x):
            if self.dim == 4: x = np.hstack([x, 0])
            _, state_hist = target.simulate(state=x, remap=enable_remap, n_steps=1)
            return state_hist[1, :] @ self.filter - x[:4]
        
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

        xi = X[:self.n_basis]
        if sigma is None: sigma = X.std(axis=0)
        k_nm = numba_k(
            X,
            xi,
            sigma,
            dim=self.dim,
        )
        
        y = np.apply_along_axis(
            simulate,
            axis=1,
            arr=X,
        )

        w, *_ = np.linalg.lstsq(k_nm, y, rcond=None)
        change = k_nm @ w

        # if enable_remap: change = np.apply_along_axis(remap, axis=1, arr=change)

        diff_train = y - change
        mse = np.sqrt(np.sum(diff_train**2)) / y.size

        self.sigma = sigma
        self.w = w
        self.xi = xi
        self.mse = mse

        return y, change

    def performAction(self, action=0.0):
        dim = self.xi.shape[1]

        if dim == 4:
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
        
        # xf = x0 @ self.filter
        x = numba_perform_action(x0, self.xi, self.sigma, self.w, dim=dim)
        
        (
            self.cart_location,
            self.cart_velocity,
            self.pole_angle,
            self.pole_velocity,
        ) = x

    def setParams(self, w, xi, sigma, mse_train=-1, mse_test=-1):
        self.w = w
        self.xi = xi
        self.sigma = sigma
        self.mse_test = mse_train
        self.mse_train = mse_test

    def setState(self, state):
        return super().setState(state)

    def resetParams(self):
        self.w = np.zeros(self.n_basis)
        self.xi = np.zeros((self.n_basis, 5))
        self.sigma = np.zeros(5)
        self.mse = 0
    
    # @classmethod
    # def _K(
    #     cls,
    #     X,
    #     XI,
    #     SIGMA,
    #     dim=5,
    # ):
    #     X = np.reshape(X, (-1, dim))
    #     XI = np.reshape(XI, (-1, dim))
    #     N = X.shape[0]
    #     M = XI.shape[0]
    #     expo = np.zeros((N, M))

    #     def se(x, xi):
    #         r = (x - xi) / SIGMA
    #         r[2] = np.sin((x[2] - xi[2]) / 2) ** 2
    #         return (np.dot(r, r)) / 2

        
    #     for i in range(X.shape[0]):
    #         for j in range(XI.shape[0]):
    #             expo[i][j] = se(
    #                 X[i, :],
    #                 XI[j, :],
    #             )

    #     return np.exp(-expo)
    

    # @classmethod
    # def _K2(
    #     cls,
    #     x,
    #     xi,
    #     sigma,
    #     dim=5,
    # ):    
    #     def helper(x, xi):        
    #         # get squared differences and substitute angle one for periodic version
    #         d = ( (x - xi) / sigma ) ** 2
    #         d[:,0] = 0
    #         d[:,2] = (np.sin( 0.5 * ( x[:,2] - xi[:,2] ) ) / sigma[2] ) ** 2
    #         # divide rows by 2 sigma and return exponential of negative sum along rows
    #         return np.exp( - 0.5 * np.sum( d, axis=1 ) )
        
    #     x = np.reshape(x, (-1, dim))
    #     xi = np.reshape(xi, (-1, dim))
    #     N = x.shape[0]
    #     M = xi.shape[0]

    #     # loop over the kernel centres and evaluate the K function across all the Xs at each
    #     k_nm = np.zeros((N, M))
    #     for i, kernel_centre in enumerate(xi):
    #         k_nm[i] = helper(x, kernel_centre[np.newaxis])
    
    

@njit
def numba_k(x, xi, sigma, dim):
    x = np.reshape(x, (-1, dim))
    xi = np.reshape(xi, (-1, dim))
    N = x.shape[0]
    M = xi.shape[0]
    expo = np.zeros((N, M))

    def se(x, xi):
        r = (x - xi) / sigma
        r[2] = np.sin((x[2] - xi[2]) / 2) ** 2
        return (np.dot(r, r)) / 2

    
    for i in range(x.shape[0]):
        for j in range(xi.shape[0]):
            expo[i][j] = se(
                x[i, :],
                xi[j, :],
            )
   
    return np.exp(-expo)


@njit 
def numba_perform_action(x, xi, sigma, w, dim):
    kernel = numba_k(
        x,
        xi,
        sigma,
        dim=dim,
    )
    change = kernel @ w
    x = x[:4] + change[0]
    return x

      



