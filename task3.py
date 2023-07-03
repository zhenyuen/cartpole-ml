from CartPole_ import CartPole
from CartPoleHat_ import CartPoleHat
from CartPoleNoisy_ import CartPoleNoisy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import default_rng
from scipy.stats import qmc, linregress
from scipy.optimize import minimize
from itertools import count


# Constants
POS_LOW = -10
POS_HIGH = 10
VEL_LOW = -10
VEL_HIGH = 10
ANG_VEL_LOW = -15
ANG_VEL_HIGH = 15
ANG_LOW = -np.pi
ANG_HIGH = np.pi

STATE0 = r"cart position, $x\ \ \ (m)$"
STATE1 = r"cart velocity, $\dot x\ \ \ (m/s)$"
STATE2 = r"pole angle, $\theta\ \ \ (rad)$"
STATE3 = r"pole velocity, $\dot \theta\ \ \ (rad/s)$"
STATE4 = r"action force, $f\ \ \ (N)$"
STATE_LABELS = [STATE0, STATE1, STATE2, STATE3, STATE4]

FORCE_LOW = -15
FORCE_HIGH = 15

P_MAX = 15
P_MIN = 0

STABLE_POS = 0
STABLE_VEL = 0
STABLE_ANG = np.pi
STABLE_ANG_VEL = 0
STABLE_EQ = np.array([STABLE_POS, STABLE_VEL, STABLE_ANG, STABLE_ANG_VEL])

UNSTABLE_POS = 0
UNSTABLE_VEL = 0
UNSTABLE_ANG = 0
UNSTABLE_ANG_VEL = 0
UNSTABLE_EQ = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL], dtype='float64')

SEED = 10


def simulate(X, cp_noisy):
    cp_noisy.setState(X)
    cp_noisy.performAction()
    return cp_noisy.getStateNoisy(loc=0, scale=1) - X # Change in state


def simulate_noisy_dyn(X, cp_noisy):
    cp_noisy.setState(X)
    cp_noisy.performActionNoisy(action=0, loc=0, scale=1)
    return cp_noisy.getStateNoisy(loc=0, scale=1) - X # Change in state


def K(X, XI, SIGMA, dim=4):
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
            expo[i][j] = se(X[i,:], XI[j,:])
    return np.exp(-expo)


def optimize_linear_policy_noisy_obs_callback(p, counter=count()):
    i = next(counter)
    print(f"Iter {i: >3}: {p}")


def noisy_obs_loss(p, cp_noisy, x0, n_steps):
    cp_noisy.setState(x0)
    total_loss = cp_noisy.loss()
    action = np.dot(p, x0)
    for _ in range(n_steps):
        cp_noisy.performAction(action)
        cp_noisy.remap_angle()

        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        total_loss += cp_noisy.loss()

        action = np.dot(p, state)
    cp_noisy.reset()
    return total_loss


def noisy_dyn_loss(p, cp_noisy, x0, n_steps):
    cp_noisy.setState(x0)
    total_loss = cp_noisy.loss()
    action = np.dot(p, x0)
    for _ in range(n_steps):
        cp_noisy.performActionNoisy(action, loc=0, scale=1)
        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        total_loss += cp_noisy.loss()

        action = np.dot(p, state)
    cp_noisy.reset()
    return total_loss


def optimize_linear_policy_noisy_obs(cp_noisy, x0, p0, n_steps, callback=optimize_linear_policy_noisy_obs_callback):
    return minimize(noisy_obs_loss, p0, args=(cp_noisy, x0, n_steps), method='Nelder-Mead', callback=callback)


def optimize_linear_policy_noisy_obs_scan_2d(cp_noisy, x0, i, j, p0, n_steps):
    # The maximum force defined in cartpole is 20.
    # This means each policy variable is contrained to between (-20, 20)
    p_min, p_max = -20, 20
    n =  8 # 40 / 5 = 8
    p_opt_hist = []

    px = np.linspace(p_min, p_max, n)
    py = np.linspace(p_min, p_max, n)

    for pi in px:
        for pj in py:
            p0[i] = pi
            p0[j] = pj
            sol = optimize_linear_policy_noisy_obs(cp_noisy, x0, p0, n_steps=n_steps, callback=None)
            p_opt_hist.append(sol.x)
    
    pz = [noisy_obs_loss(p, cp_noisy, x0, n_steps) for p in p_opt_hist]
    pz = np.array(pz)
    return px, py, pz, p_opt_hist


def optimize_linear_policy_noisy_obs_scan_2d_plotter(x, y, z, i, j, fig, ax, color='white'): 
    d = x.shape[0]
    z = np.reshape(z, newshape=(d, d))
    c1 = ax.contour(x, y, z, colors=color, levels=5)
    cntr = ax.contourf(x, y, z, levels=20)
    fig.colorbar(cntr, ax=ax)
    ax.set_title("Total loss")
    ax.set(xlabel=f"$p_{i}$", ylabel=f"$p_{j}$")
    ax.clabel(c1, c1.levels, inline=True, colors=color)


def optimize_linear_policy_noisy_obs_scan_3d(cp_noisy, x0, i, j, k, p0, n_steps):
    # The maximum force defined in cartpole is 20.
    # This means each policy variable is contrained to between (-20, 20)
    p_min, p_max = -20, 20
    n = 4 # 40 / 10 = 4
    p_opt_hist = []

    pw = np.linspace(p_min, p_max, n)
    px = np.linspace(p_min, p_max, n)
    py = np.linspace(p_min, p_max, n)

    for pi in pw:
        for pj in px:
            for pk in py:
                p0[i] = pi
                p0[j] = pj
                p0[k] = pk
                sol = optimize_linear_policy_noisy_obs(cp_noisy, x0, p0, n_steps=n_steps, callback=None)
                p_opt_hist.append(sol.x)
        
    pz = [noisy_obs_loss(p, cp_noisy, x0, n_steps) for p in p_opt_hist]
    pz = np.array(pz)
    return pw, px, py, pz, p_opt_hist


def optimize_linear_policy_noisy_obs_scan_3d_plotter(w, x, y, z, i, j, k, fig, ax, color='white'): 
    ww, xx, yy = np.meshgrid(w, x, y)

    sc = ax.scatter(ww, xx, yy, c=z, vmin=z.min(), vmax=z.max(), cmap=plt.cm.get_cmap('jet'), s=70)
    fig.colorbar(sc, ax=ax)
    ax.set_title("Total loss")
    ax.set(xlabel=f"$p_{i}$", ylabel=f"$p_{j}$",  zlabel=f"$p_{k}$")

    opt_p = (0, 0, 0, 0)
    opt_loss = float('inf')

    d = x.shape[0]
    zf = np.reshape(z, newshape=(d, d, d)) 
    
    for i in range(len(w)):
        for j in range(len(x)):
            for k in range(len(y)):
                if zf[i, j, k] < opt_loss:
                    opt_loss = zf[i, j, k]
                    opt_p = (1, w[i], x[j], y[k])

    print(f"Optimal P: {opt_p}")


def optimize_linear_policy_noisy_dyn(cp_noisy, x0, p0, n_steps, callback=optimize_linear_policy_noisy_obs_callback):
    return minimize(noisy_dyn_loss, p0, args=(cp_noisy, x0, n_steps), method='Nelder-Mead', callback=callback)




def get_rollout_points():
        return (
        (0, 0.01, np.pi, 0.01), # stable equilibrium, simple oscillations
        (0, 2, np.pi+0.1, 2), # stable equilibrium
        (0, 2, np.pi+0.1, 2), # stable equilibrium
        (0, 2, np.pi+0.1, 14), # Unstable equilibrium, large pole velocity
        (0, 14, np.pi+0.1, 2), # Unstable equilibrium, large pole velocity
        (0, 14, np.pi+0.1, 14), # Unstable equilibrium, large pole and cart velocity
        (0, 8, np.pi+0.1, 8), # Unstable equilibrium, large pole and cart velocity, but no complete oscillation
        (0, 2, np.pi / 2, 2), # Halfway point,
        (0, 0.01, 0.1, 0.01), # Unstable equilibrium, no initial velocities
        (0, 2, 0.1, 2), # Unstable equilibrium
        (0, 2, 0.1, 2), # Unstable equilibrium
        (0, 2, 0.1, 14), # Unstable equilibrium, large pole velocity
        (0, 14, 0.1, 2), # Unstable equilibrium, large pole velocity
        (0, 14, 0.1, 14), # Unstable equilibrium, large pole and cart velocity
        (0, 8, 0.1, 8), # Unstable equilibrium, large pole and cart velocity, but no complete oscillation
        (0, 8, 0.1, 8) # Halfway point, large pole and cart velocity
    )


def t3_1_rollout_scan_cart_velocity(cp_noisy, axs, fig):
    steps = np.arange(50 + 1)
    l = 5
    
    colors = plt.cm.jet(np.linspace(0, 1, l))

    for i, vel in enumerate(np.linspace(VEL_LOW, VEL_HIGH, l)):
        x0 = np.array([STABLE_POS, vel, STABLE_ANG, STABLE_ANG_VEL])
        cp_noisy.setState(x0)

        x = x0[None, ...]
        
        for _ in steps[1: ]:
            x = np.vstack([x, cp_noisy.getStateNoisy(loc=0, scale=1)])
            cp_noisy.performAction()

        t3_1_rollout_scan_cart_velocity_plotter(steps, x, fig=fig, axs=axs, label=f"$\dot x$={vel:.2f}", color=colors[i])
        cp_noisy.reset()


def t3_1_rollout_scan_cart_velocity_plotter(x, y, fig, axs, color=None, label=""):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle='solid')
    axs[0,1].plot(x, y[:, 1], color=color, linestyle='solid')
    axs[1,0].plot(x, y[:, 2], color=color, linestyle='solid')
    axs[1,1].plot(x, y[:, 3], color=color, linestyle='solid')

    #Set titles
    ylabels = STATE_LABELS
    xlabels = [r"time $(s)$"] * 4

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()


def t3_1_rollout_scan_pole_velocity(cp_noisy, axs, fig):
    steps = np.arange(50 + 1)
    l = 5
    
    colors = plt.cm.jet(np.linspace(0, 1, l))

    for i, ang_vel in enumerate(np.linspace(ANG_VEL_LOW, ANG_VEL_HIGH, l)):
        x0 = np.array([STABLE_POS, STABLE_VEL, STABLE_ANG, ang_vel])
        cp_noisy.setState(x0)

        x = x0[None, ...]
        
        for _ in steps[1: ]:
            x = np.vstack([x, cp_noisy.getStateNoisy(loc=0, scale=1)])
            cp_noisy.performAction()

        t3_1_rollout_scan_pole_velocity_plotter(steps, x, fig=fig, axs=axs, label=f"$\dot \theta$={ang_vel:.2f}", color=colors[i])
        cp_noisy.reset()


def t3_1_rollout_scan_pole_velocity_plotter(x, y, fig, axs, color=None, label=""):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle='solid')
    axs[0,1].plot(x, y[:, 1], color=color, linestyle='solid')
    axs[1,0].plot(x, y[:, 2], color=color, linestyle='solid')
    axs[1,1].plot(x, y[:, 3], color=color, linestyle='solid')

    #Set titles
    ylabels = STATE_LABELS
    xlabels = [r"time $(s)$"] * 4

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()


def t3_1_rollout_scan_pole_velocity(cp_noisy, axs, fig):
    steps = np.arange(50 + 1)
   
    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))

    for i, ang_vel in enumerate(np.linspace(ANG_VEL_LOW, ANG_VEL_HIGH, l)):
        x0 = np.array([STABLE_POS, STABLE_VEL, STABLE_ANG, ang_vel])
        cp_noisy.setState(x0)

        x = x0[None, ...]
        
        for _ in steps[1: ]:
            x = np.vstack([x, cp_noisy.getStateNoisy(loc=0, scale=1)])
            cp_noisy.performAction()

        t3_1_rollout_scan_pole_velocity_plotter(steps, x, fig=fig, axs=axs, label=f"$\dot \theta$={ang_vel:.2f}", color=colors[i])
        cp_noisy.reset()


def t3_1_rollout_scan_pole_velocity_plotter(x, y, fig, axs, color=None, label=""):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle='solid')
    axs[0,1].plot(x, y[:, 1], color=color, linestyle='solid')
    axs[1,0].plot(x, y[:, 2], color=color, linestyle='solid')
    axs[1,1].plot(x, y[:, 3], color=color, linestyle='solid')

    #Set titles
    ylabels = STATE_LABELS
    xlabels = [r"time $(s)$"] * 4

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()


def t3_1_fit_linear_model(cp_noisy, axs, fig, n=11):
    sampler = qmc.Sobol(d=4, seed=10)
    X = sampler.random_base2(n)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW

    Y = np.apply_along_axis(simulate, axis=1, arr=X, cp_noisy=cp_noisy)
    W, residuals, rank, s = np.linalg.lstsq(X, Y)

    print(f"SE: {residuals}")

    X_hat = X @ W + X

    t3_1_fit_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W, residuals


def t3_1_fit_linear_model_random(cp_noisy, axs, fig, n=11):
    rng = default_rng(SEED)
    size = int(2 ** n)
    X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
    X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    X = np.vstack([X0, X1, X2, X3]).T

    Y = np.apply_along_axis(simulate, axis=1, arr=X, cp_noisy=cp_noisy)
    W, residuals, rank, s = np.linalg.lstsq(X, Y)

    print(f"SE: {residuals}")

    X_hat = X @ W + X

    t3_1_fit_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W, residuals


def t3_1_fit_linear_model_random_manual_lstsq(cp_noisy, axs, fig, n=11):
    rng = default_rng(SEED)
    size = int(2 ** n)
    X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
    X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    X = np.vstack([X0, X1, X2, X3]).T

    Y = np.apply_along_axis(simulate, axis=1, arr=X, cp_noisy=cp_noisy)
    W = np.linalg.inv(X.T @ X) @ X.T @ Y

    X_hat = X @ W + X

    t3_1_fit_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W


def t3_1_fit_linear_model_plotter(x, x_hat, fig, axs, diag=True):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], x_hat[:, 0])
    axs[0,1].scatter(x[:, 1], x_hat[:, 1])
    axs[1,0].scatter(x[:, 2], x_hat[:, 2])
    axs[1,1].scatter(x[:, 3], x_hat[:, 3])

    #Set titles
    ylabels = [f"predicted {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_1_fit_non_linear_model(cp_noisy, axs, fig, n=11, m=5):
    n_basis = 2 ** m * 10

    sampler = qmc.Sobol(d=4, seed=10)
    X = sampler.random_base2(n)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    XI = X[:n_basis]
    
    SIGMA = X.std(axis=0)
    K_NM = K(X, XI, SIGMA)

    Y = np.apply_along_axis(simulate, axis=1, arr=X, cp_noisy=cp_noisy)
    W, residuals, rank, s = np.linalg.lstsq(K_NM, Y)
    print(f"SE: {residuals}")

    X_hat = K_NM @ W + X

    t3_1_fit_non_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W, XI, K_NM, SIGMA, residuals


def t3_1_fit_non_linear_model_plotter(x, x_hat, fig, axs, diag=True):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], x_hat[:, 0])
    axs[0,1].scatter(x[:, 1], x_hat[:, 1])
    axs[1,0].scatter(x[:, 2], x_hat[:, 2])
    axs[1,1].scatter(x[:, 3], x_hat[:, 3])

    #Set titles
    ylabels = [f"predicted {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_1_fit_linear_model_change_plotter(x, y, w, fig, axs, diag=True):
    y_hat = x @ w

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(y[:, 0], y_hat[:, 0])
    axs[0,1].scatter(y[:, 1], y_hat[:, 1])
    axs[1,0].scatter(y[:, 2], y_hat[:, 2])
    axs[1,1].scatter(y[:, 3], y_hat[:, 3])

    #Set titles
    ylabels = [f"predicted change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_1_fit_non_linear_model_change_plotter(x, y, w, xi, k_nm, sigma, fig, axs, diag=True):
    y_hat = k_nm @ w

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(y[:, 0], y_hat[:, 0])
    axs[0,1].scatter(y[:, 1], y_hat[:, 1])
    axs[1,0].scatter(y[:, 2], y_hat[:, 2])
    axs[1,1].scatter(y[:, 3], y_hat[:, 3])

    #Set titles
    ylabels = [f"predicted change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_1_linear_model_linear_policy_rollout(cp_noisy, p, fig, axs, n_steps=50):
    x0 = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL])
    cp_noisy.setState(x0)

    x = x0[None, ...]
    action_hist = []

    steps = np.arange(n_steps + 1)

    p = [1, 1, 1, 1]
    action = np.dot(p, x0)
    action_hist.append(action)
    for _ in steps[1: ]:
        cp_noisy.performAction(action)
        cp_noisy.remap_angle()

        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        x = np.vstack([x, state])
        
        action = np.dot(p, state)
        action_hist.append(action)

    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))

    action_hist = np.array(action_hist)

    t3_1_linear_model_linear_policy_rollout_plotter(steps, x, action_hist, fig, axs, colors)
    cp_noisy.reset()


def t3_1_linear_model_linear_policy_rollout_plotter(x, y, a, fig, axs, colors):
    labels = STATE_LABELS
    axs[0].plot(x, y[:, 0], color=colors[0], label=labels[0], linestyle='solid')
    axs[0].plot(x, y[:, 1], color=colors[1], label=labels[1], linestyle='solid')
    axs[0].plot(x, y[:, 2], color=colors[2], label=labels[2], linestyle='solid')
    axs[0].plot(x, y[:, 3], color=colors[3], label=labels[3], linestyle='solid')
    axs[0].plot(x, a, color=colors[4], label=labels[4], linestyle='solid')

    axs[1].plot(x, np.abs(y[:, 0]), color=colors[0], linestyle='solid') # Take absolute as we are only interested in magnitude of devations
    axs[1].plot(x, np.abs(y[:, 1]), color=colors[1], linestyle='solid')
    axs[1].plot(x, np.abs(y[:, 2]), color=colors[2], linestyle='solid')
    axs[1].plot(x, np.abs(y[:, 3]), color=colors[3], linestyle='solid')
    axs[1].plot(x, np.abs(a), color=colors[4], linestyle='solid')
    
    #Set titles
    ylabel = "response"
    xlabel = r"time $(s)$"

    axs[0].set(xlabel=xlabel, ylabel=ylabel)
    axs[0].grid()
    axs[1].set(xlabel=xlabel, ylabel=ylabel)
    axs[1].grid()

    axs[1].set_yscale('log')


def t3_1_linear_model_linear_policy_optimize(cp_noisy, p, fig, axs):
    x0 = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL])
    cp_noisy.setState(x0)

    x = x0[None, ...]
    action_hist = []

    steps = np.arange(50 + 1)

    p = [1, 1, 1, 1]
    action = np.dot(p, x0)
    action_hist.append(action)
    for _ in steps[1: ]:
        cp_noisy.performAction(action)
        cp_noisy.remap_angle()

        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        x = np.vstack([x, state])
        
        action = np.dot(p, state)
        action_hist.append(action)

    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))


def t3_1_non_linear_model_linear_policy(cp_noisy, cp_hat, x, y, w, xi, k_nm, sigma, fig, axs, diag=True):
    pass


def t3_2_fit_linear_model(cp_noisy, axs, fig, n=11):
    sampler = qmc.Sobol(d=4, seed=10)
    X = sampler.random_base2(n)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW

    Y = np.apply_along_axis(simulate_noisy_dyn, axis=1, arr=X, cp_noisy=cp_noisy)
    W, residuals, rank, s = np.linalg.lstsq(X, Y)

    print(f"SE: {residuals}")

    X_hat = X @ W + X

    t3_1_fit_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W, residuals


def t3_2_fit_non_linear_model(cp_noisy, axs, fig, n=11, m=5):
    n_basis = 2 ** m * 10

    sampler = qmc.Sobol(d=4, seed=10)
    X = sampler.random_base2(n)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    XI = X[:n_basis]
    
    SIGMA = X.std(axis=0)
    K_NM = K(X, XI, SIGMA)

    Y = np.apply_along_axis(simulate_noisy_dyn, axis=1, arr=X, cp_noisy=cp_noisy)
    W, residuals, rank, s = np.linalg.lstsq(K_NM, Y)
    print(f"SE: {residuals}")

    X_hat = K_NM @ W + X

    t3_1_fit_non_linear_model_plotter(X, X_hat, fig=fig, axs=axs)

    return X ,Y, W, XI, K_NM, SIGMA, residuals


def t3_2_fit_linear_model_change_plotter(x, y, w, fig, axs, diag=True):
    y_hat = x @ w

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(y[:, 0], y_hat[:, 0])
    axs[0,1].scatter(y[:, 1], y_hat[:, 1])
    axs[1,0].scatter(y[:, 2], y_hat[:, 2])
    axs[1,1].scatter(y[:, 3], y_hat[:, 3])

    #Set titles
    ylabels = [f"predicted change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_2_fit_non_linear_model_change_plotter(x, y, w, xi, k_nm, sigma, fig, axs, diag=True):
    y_hat = k_nm @ w

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(y[:, 0], y_hat[:, 0])
    axs[0,1].scatter(y[:, 1], y_hat[:, 1])
    axs[1,0].scatter(y[:, 2], y_hat[:, 2])
    axs[1,1].scatter(y[:, 3], y_hat[:, 3])

    #Set titles
    ylabels = [f"predicted change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual change $\Delta$ in {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_2_linear_model_linear_policy_rollout(cp_noisy, p, fig, axs, n_steps=50):
    x0 = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL])
    cp_noisy.setState(x0)

    x = x0[None, ...]
    action_hist = []

    steps = np.arange(n_steps + 1)

    p = [1, 1, 1, 1]
    action = np.dot(p, x0)
    action_hist.append(action)
    for _ in steps[1: ]:
        cp_noisy.performActionNoisy(action, loc=0, scale=1)
        cp_noisy.remap_angle()

        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        x = np.vstack([x, state])
        
        action = np.dot(p, state)
        action_hist.append(action)

    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))

    action_hist = np.array(action_hist)

    t3_2_linear_model_linear_policy_rollout_plotter(steps, x, action_hist, fig, axs, colors)
    cp_noisy.reset()


def t3_2_linear_model_linear_policy_rollout_plotter(x, y, a, fig, axs, colors):
    labels = STATE_LABELS
    axs[0].plot(x, y[:, 0], color=colors[0], label=labels[0], linestyle='solid')
    axs[0].plot(x, y[:, 1], color=colors[1], label=labels[1], linestyle='solid')
    axs[0].plot(x, y[:, 2], color=colors[2], label=labels[2], linestyle='solid')
    axs[0].plot(x, y[:, 3], color=colors[3], label=labels[3], linestyle='solid')
    axs[0].plot(x, a, color=colors[4], label=labels[4], linestyle='solid')

    axs[1].plot(x, np.abs(y[:, 0]), color=colors[0], linestyle='solid') # Take absolute as we are only interested in magnitude of devations
    axs[1].plot(x, np.abs(y[:, 1]), color=colors[1], linestyle='solid')
    axs[1].plot(x, np.abs(y[:, 2]), color=colors[2], linestyle='solid')
    axs[1].plot(x, np.abs(y[:, 3]), color=colors[3], linestyle='solid')
    axs[1].plot(x, np.abs(a), color=colors[4], linestyle='solid')
    
    #Set titles
    ylabel = "response"
    xlabel = r"time $(s)$"

    axs[0].set(xlabel=xlabel, ylabel=ylabel)
    axs[0].grid()
    axs[1].set(xlabel=xlabel, ylabel=ylabel)
    axs[1].grid()

    axs[1].set_yscale('log')


def t3_2_linear_model_linear_policy_optimize(cp_noisy, p, fig, axs):
    x0 = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL])
    cp_noisy.setState(x0)

    x = x0[None, ...]
    action_hist = []

    steps = np.arange(50 + 1)

    p = [1, 1, 1, 1]
    action = np.dot(p, x0)
    action_hist.append(action)
    for _ in steps[1: ]:
        cp_noisy.performActionNoisy(action, loc=0, scale=1)
        cp_noisy.remap_angle()

        state = cp_noisy.getStateNoisy(loc=0, scale=1)
        x = np.vstack([x, state])
        
        action = np.dot(p, state)
        action_hist.append(action)

    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))


# def t3_1_non_linear_model_linear_policy(cp_noisy, cp_hat, x, y, w, xi, k_nm, sigma, fig, axs, diag=True):
#     pass


