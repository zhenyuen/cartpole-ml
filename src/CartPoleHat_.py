from CartPole_ import CartPole
import numpy as np
from numpy.random import default_rng
from scipy.stats import qmc, linregress


POS_LOW = -10
POS_HIGH = 10
VEL_LOW = -10
VEL_HIGH = 10
ANG_VEL_LOW = -15
ANG_VEL_HIGH = 15
ANG_LOW = -np.pi
ANG_HIGH = np.pi
FORCE_LOW = -15
FORCE_HIGH = 15

P_MAX = 15
P_MIN = 0

STATE0 = r"cart position, $x\hspace{2mm}(m)$"
STATE1 = r"cart velocity, $\dot{x}\hspace{2mm}(m/s)$"
STATE2 = r"pole angle, $\theta\hspace{2mm}(rad)$"
STATE3 = r"pole velocity, $\dot{\theta}\hspace{2mm}(rad/s)$"
STATE4 = r"action force, $F\hspace{2mm}(N)$"
STATE_LABELS = [STATE0, STATE1, STATE2, STATE3, STATE4]


STABLE_POS = 0
STABLE_VEL = 0
STABLE_ANG = np.pi
STABLE_ANG_VEL = 0

UNSTABLE_POS = 0
UNSTABLE_VEL = 0
UNSTABLE_ANG = 0
UNSTABLE_ANG_VEL = 0

class CartPoleHat(CartPole):
    def __init__(self, visual=False):
        super().__init__(visual)
        self.w = None
        self.xi = None
        self.sigma = None
        self.mse_test = None
        self.mse_train = None
        self.state_hist = []

    def train(self, n, m):
        n_data = 2**n
        n_basis = 2**m * 10
        self._compute_weights_tts(
            n_data,
            n_basis,
            basis_placement="sobol",
            train_split=0.8,
        )

    def performAction(self, action=None):
        if action is not None:
            i_state = np.array(
                [
                    self.cart_location,
                    self.cart_velocity,
                    self.pole_angle,
                    self.pole_velocity,
                    action,
                ]
            )
            kernel = self._K(
                i_state,
                self.xi,
                self.sigma,
                dim=5,
            )
        else:
            i_state = np.array(
                [
                    self.cart_location,
                    self.cart_velocity,
                    self.pole_angle,
                    self.pole_velocity,
                ]
            )
            kernel = self._K(
                i_state,
                self.xi,
                self.sigma,
                dim=4,
            )
        change = kernel @ self.w
        f_state = i_state[:4] + change
        self.state_hist.append(f_state)
        (
            self.cart_location,
            self.cart_velocity,
            self.pole_angle,
            self.pole_velocity,
        ) = f_state[0]

    def setTrainedParams(self, w, xi, sigma, mse_train=-1, mse_test=-1):
        self.w = w
        self.xi = xi
        self.sigma = sigma
        self.mse_test = mse_train
        self.mse_train = mse_test

    def setState(self, state):
        self.state_hist.append(state)
        return super().setState(state)

    def reset(self):
        self.state_hist = []
        return super().reset()

    def getStateHist(self):
        return self.state_hist

    def _K(
        self,
        X,
        XI,
        SIGMA,
        dim=5,
    ):
        X = np.reshape(X, (-1, dim))
        XI = np.reshape(XI, (-1, dim))
        N = X.shape[0]
        M = XI.shape[0]
        expo = np.zeros((N, M))

        def se(x, xi):
            r = (x - xi) / SIGMA
            r[2] = np.sin((x[2] - xi[2]) / 2) ** 2
            return (np.dot(r, r)) / 2

        for i in range(X.shape[0]):
            for j in range(XI.shape[0]):
                expo[i][j] = se(
                    X[i, :],
                    XI[j, :],
                )
        return np.exp(-expo)

    def _compute_weights_tts_with_action(
        self,
        N,
        M,
        basis_placement="random",
        train_split=1.0,
    ):
        def simulate(X):
            state = X[:4]
            cp = CartPole(False)
            cp.setState(state)
            cp.performAction(X[-1])
            return cp.getState() - state  # Change in state

        exponent = int(np.ceil(np.log2(N)))
        seed = 10
        split = int(N * train_split)

        if basis_placement == "random":
            rng = default_rng(seed)
            size = int(2**exponent)
            X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
            X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
            X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
            X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
            X4 = rng.random(size) * (FORCE_HIGH - FORCE_LOW) + FORCE_LOW
            X = np.vstack(
                [
                    X0,
                    X1,
                    X2,
                    X3,
                    X4,
                ]
            ).T

            X_train = X[:split, ...]
            X_test = X[split:, ...]
            XI = rng.choice(
                X_train,
                size=M,
                replace=False,
            )

        elif basis_placement == "sobol":
            sampler = qmc.Sobol(
                d=5,
                seed=seed,
            )
            X = sampler.random_base2(m=exponent)
            X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
            X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
            X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
            X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
            X[:, 4] = X[:, 4] * (FORCE_HIGH - FORCE_LOW) + FORCE_LOW

            X_train = X[:split, ...]
            X_test = X[split:, ...] if split != N else None
            XI = X_train[:M]

        SIGMA = X_train.std(axis=0)
        K_NM = self.k_func(
            X_train,
            XI,
            SIGMA,
            dim=5,
        )

        Y_train = np.apply_along_axis(
            simulate,
            axis=1,
            arr=X_train,
        )
        Y_test = (
            np.apply_along_axis(
                simulate,
                axis=1,
                arr=X_test,
            )
            if split != N
            else None
        )

        (
            W,
            _,
            _,
            _,
        ) = np.linalg.lstsq(K_NM, Y_train)

        diff_train = Y_train - K_NM @ W
        mse_train = np.sqrt(np.sum(diff_train**2)) / Y_train.size

        if split != N:
            diff_test = (
                Y_test
                - self.k_func(
                    X_test,
                    XI,
                    SIGMA,
                )
                @ W
            )
            mse_test = np.sqrt(np.sum(diff_test**2)) / Y_test.size
        else:
            mse_test = 0

        self.sigma = SIGMA
        self.w = W
        self.xi = XI
        self.mse_train = mse_train
        self.mse_test = mse_test



