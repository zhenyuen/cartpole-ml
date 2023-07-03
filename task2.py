from utilities import *
from numpy.random import default_rng
from scipy.stats import qmc, linregress
from scipy.optimize import minimize
from CartPole_ import *
from CartPoleHat_ import CartPoleHat

STATE0 = r"cart position, $x\hspace{2mm}(m)$"
STATE1 = r"cart velocity, $\dot{x}\hspace{2mm}(m/s)$"
STATE2 = r"pole angle, $\theta\hspace{2mm}(rad)$"
STATE3 = r"pole velocity, $\dot{\theta}\hspace{2mm}(rad/s)$"
STATE4 = r"action force, $f\hspace{2mm}(N)$"
STATE_LABELS = [STATE0, STATE1, STATE2, STATE3, STATE4]

FORCE_LOW = -15
FORCE_HIGH = 15

P_MAX = 15
P_MIN = 0

ROLLOUT_INITIAL_CONTIDITIONS = (
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

def get_ranges(n):
    POS_RANGE = np.linspace(POS_LOW, POS_HIGH, n)
    VEL_RANGE = np.linspace(VEL_LOW, VEL_HIGH, n)
    ANG_RANGE = np.linspace(ANG_LOW, ANG_HIGH, n)
    ANG_VEL_RANGE = np.linspace(ANG_VEL_LOW, ANG_VEL_HIGH, n)
    FORCE_RANGE = np.linspace(FORCE_LOW, FORCE_HIGH, n)
    return POS_RANGE, VEL_RANGE, ANG_RANGE, ANG_VEL_RANGE, FORCE_RANGE


def K(X, XI, SIGMA, dim=4):
    X = np.reshape(X, (-1,dim))
    XI = np.reshape(XI, (-1,dim))
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


def simulate(X):
    cp = CartPole(False)
    cp.setState(X)
    cp.performAction()
    return cp.getState() - X


def compute_weights(N, M, basis_placement='random', k_func=K):
    def simulate(X):
        cp = CartPole(False)
        cp.setState(X)
        cp.performAction()
        return cp.getState() - X # Change in state

    exponent = int(np.ceil(np.log2(N)))
    seed = 10

    print(f"Samples drawn: {2 ** exponent}")
    if basis_placement == 'random':
        rng = default_rng(seed)
        size = int(2 ** exponent)
        X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
        X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        X = np.vstack([X0, X1, X2, X3]).T
        XI = rng.choice(X, size=M, replace=False)

    elif basis_placement == 'sobol':
        sampler = qmc.Sobol(d=4, seed=seed)
        X = sampler.random_base2(m=exponent)
        X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
        X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        XI = X[:M]
    
    SIGMA = X.std(axis=0)
    K_NM = k_func(X, XI, SIGMA)

    Y = np.apply_along_axis(simulate, axis=1, arr=X)
    W, residuals, rank, s = np.linalg.lstsq(K_NM, Y)
    print(f"SE: {residuals}")

    return X, XI,Y, W, K_NM


def compute_weights_tts(N, M, basis_placement='random', k_func=K, train_split=1.0):
    def simulate(X):
        cp = CartPole(False)
        cp.setState(X)
        cp.performAction()
        return cp.getState() - X # Change in state

    exponent = int(np.ceil(np.log2(N)))
    seed = 10
    split = int(N * train_split)

    print(f"Samples drawn: {2 ** exponent}")
    if basis_placement == 'random':
        rng = default_rng(seed)
        size = int(2 ** exponent)
        X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
        X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        X = np.vstack([X0, X1, X2, X3]).T
        
        X_train = X[:split, ...]
        X_test = X[split:, ...]
        XI = rng.choice(X_train, size=M, replace=False)

    elif basis_placement == 'sobol':
        sampler = qmc.Sobol(d=4, seed=seed)
        X = sampler.random_base2(m=exponent)
        X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
        X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        
        X_train = X[:split, ...]
        X_test = X[split:, ...] if split != N else None
        XI = X_train[:M]
    
    SIGMA = X_train.std(axis=0)
    K_NM = k_func(X_train, XI, SIGMA)
    
    
    Y_train = np.apply_along_axis(simulate, axis=1, arr=X_train)
    Y_test = np.apply_along_axis(simulate, axis=1, arr=X_test) if split != N else None
    
    W, residuals, rank, s = np.linalg.lstsq(K_NM, Y_train)
    print(f"SE: {residuals}")
    
    diff_train = Y_train - K_NM @ W
    mse_train = np.sqrt(np.sum(diff_train ** 2)) / Y_train.size
    
    diff_test = Y_test - k_func(X_test, XI, SIGMA) @ W
    mse_test = np.sqrt(np.sum(diff_test ** 2)) / Y_test.size
    

    return X_train, XI, Y_train, X_test, Y_test, W, K_NM, mse_train, mse_test

    
def compute_weights_tts_with_action(N, M, basis_placement='random', k_func=K, train_split=1.0):
    def simulate(X):
        state = X[:4]
        cp = CartPole(False)
        cp.setState(state)
        cp.performAction(X[-1])
        return cp.getState() - state # Change in state

    exponent = int(np.ceil(np.log2(N)))
    seed = 10
    split = int(N * train_split)

    print(f"Samples drawn: {2 ** exponent}")
    if basis_placement == 'random':
        rng = default_rng(seed)
        size = int(2 ** exponent)
        X0 = rng.random(size) * (POS_HIGH - POS_LOW) + POS_LOW
        X1 = rng.random(size) * (VEL_HIGH - VEL_LOW) + VEL_LOW
        X2 = rng.random(size) * (ANG_HIGH - ANG_LOW) + ANG_LOW
        X3 = rng.random(size) * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
        X4 = rng.random(size) * (FORCE_HIGH - FORCE_LOW) + FORCE_LOW
        X = np.vstack([X0, X1, X2, X3, X4]).T
        
        X_train = X[:split, ...]
        X_test = X[split:, ...]
        XI = rng.choice(X_train, size=M, replace=False)

    elif basis_placement == 'sobol':
        sampler = qmc.Sobol(d=5, seed=seed)
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
    K_NM = k_func(X_train, XI, SIGMA, dim=5)
    print(K_NM.shape)
    

    Y_train = np.apply_along_axis(simulate, axis=1, arr=X_train)
    Y_test = np.apply_along_axis(simulate, axis=1, arr=X_test) if split != N else None
    
    W, residuals, rank, s = np.linalg.lstsq(K_NM, Y_train)
    print(W.shape)
    print(f"SE: {residuals}")
    
    diff_train = Y_train - K_NM @ W
    mse_train = np.sqrt(np.sum(diff_train ** 2)) / Y_train.size

    if split != N:
        diff_test = Y_test - k_func(X_test, XI, SIGMA) @ W
        mse_test = np.sqrt(np.sum(diff_test ** 2)) / Y_test.size
    else:
        mse_test = 0
    
    return X_train, XI, Y_train, X_test, Y_test, W, K_NM, mse_train, mse_test


def t2_1_roll_out(x, y, y_hat, fig, axs, color=None, label="", xlabel="", ylabel="", xlim=None, ylim=None):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle='solid')
    axs[0,1].plot(x, y[:, 1], color=color, linestyle='solid')
    axs[1,0].plot(x, y[:, 2], color=color, linestyle='solid')
    axs[1,1].plot(x, y[:, 3], color=color, linestyle='solid')

    axs[0,0].plot(x, y_hat[:, 0], color=color, linestyle='dashed')
    axs[0,1].plot(x, y_hat[:, 1], color=color, linestyle='dashed')
    axs[1,0].plot(x, y_hat[:, 2], color=color, linestyle='dashed')
    axs[1,1].plot(x, y_hat[:, 3], color=color, linestyle='dashed')

    #Set titles
    ylabels = [ylabel] * 4 if ylabel else [STATE0, STATE1, STATE2, STATE3]
    xlabels = [xlabel] * 4 if xlabel else [STATE0, STATE1, STATE2, STATE3]

    axs[0,0].set(xlabel=xlabels[0])
    axs[0,0].set(ylabel=ylabels[0])
    axs[0,1].set(xlabel=xlabels[1])
    axs[0,1].set(ylabel=ylabels[1])
    axs[1,0].set(xlabel=xlabels[2])
    axs[1,0].set(ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3])
    axs[1, 1].set(ylabel=ylabels[3])

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0, 0].set(xlim=xlim, ylim=ylim)
    axs[0, 1].set(xlim=xlim, ylim=ylim)
    axs[1, 0].set(xlim=xlim, ylim=ylim)
    axs[1, 1].set(xlim=xlim, ylim=ylim)

    thresholds = abs(y_hat - y) > 3 * np.std(y, axis=0)
    div_x = np.zeros(4)
    div_y = np.zeros(4)
    div_osc = np.zeros(4)

    # Period of oscillation is 3

    for i in range(4):
        for pos, div in enumerate(thresholds[:, i]):
            div_x[i] = x[pos]
            div_y[i] = y[pos, i]
            div_osc[i] = pos / 3
            if div: break
    
    # Mark point of divergence
    axs[0,0].scatter(div_x[0], div_y[0], color=color, s=60)
    axs[0,1].scatter(div_x[1], div_y[1], color=color, s=60)
    axs[1,0].scatter(div_x[2], div_y[2], color=color, s=60)
    axs[1,1].scatter(div_x[3], div_y[3], color=color, s=60)

    div = [f"({x:.2f}, {y:.2f}, {o:.2f})" for x, y, o in zip(div_x, div_y, div_osc)] # div_osc is the number of oscillations
    
    print(f"Point of divergence:", ", ".join(div))



def t2_3_roll_out(x, y, fig, ax, color=None, label=None, title=None):
    ax.plot(x, y, color=color, label=label, linestyle='solid')
    ax.set_xlabel(r'simulation steps $(\times 50)$')
    ax.set_ylabel(r'loss')
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()


def t2_3_opt_p(initial_state, initial_p, steps=14):
    def loss(p):
        cp = CartPole(False)
        cp.sim_steps = 1
        cp.setState(initial_state)
        total_loss = 0
        action = np.dot(p, initial_state)
        for _ in range(steps):
            cp.remap_angle()
            cp.performAction(action)
            state = cp.getState()

            if np.any( np.abs(state) > np.array([30, 30, 30, 30]) ):
                return steps
            
            action = np.dot(p, state)
            total_loss += cp.loss()
        return total_loss
    
    return minimize(loss, initial_p, method='Nelder-Mead', callback=print)


def t2_4_opt_p(cp_hat, initial_state, initial_p, steps=10):
    def loss(p):
        print(p)
        cp_hat.setState(initial_state)
        total_loss = cp_hat.loss()
        action = np.dot(p, initial_state)
        for _ in range(steps):
            cp_hat.remap_angle()
            cp_hat.performAction(action)
            state = cp_hat.getState()
            action = np.dot(p, state)
            total_loss += cp_hat.loss()
        cp_hat.reset()
        return total_loss
    
    return minimize(loss, initial_p, method='Nelder-Mead')


def t2_3_opt_p_custom(initial_state, initial_p, steps=10):
    def loss(p):
        cp = CartPole(False)
        cp.setState(initial_state)
        total_loss = np.log(cp.loss())
        action = np.dot(p, initial_state)
        for _ in range(steps):
            cp.remap_angle()
            cp.performAction(action)
            state = cp.getState()
            action = np.dot(p, state)
            total_loss += np.log(cp.loss())
        return cp.loss()
    
    return minimize(loss, initial_p, method='Nelder-Mead')


def t2_3_opt_p_individual(initial_state, initial_p, steps=10, fixed_p=[1,1,1,1], vary=0):
    def loss(p):
        fixed_p[vary] = p
        print(fixed_p)
        cp = CartPole(False)
        cp.setState(initial_state)
        total_loss = cp.loss()
        action = np.dot(fixed_p, initial_state)
        for _ in range(steps):
            cp.remap_angle()
            cp.performAction(action)
            state = cp.getState()
            action = np.dot(fixed_p, state)
            total_loss += cp.loss()
        return total_loss
    
    return minimize(loss, initial_p, method='Nelder-Mead')


def t2_3_opt_p_with_period(initial_state, initial_p, std_f=1, oscillations=2, steps=10):
    def loss(p):
        p_ang = []
        total_loss = 0
        cp = CartPole(False)
        cp.setState(initial_state)
        loss_hist = []
        action = np.dot(p, initial_state)
        p_ang.append(initial_state[2])
        j = steps

        for _ in range(steps):
            cp.performAction(action)
            state = cp.getState()
            action = np.dot(p, state)
            loss_hist.append(cp.loss())
            p_ang.append(initial_state[2])

        # Count the current number of periods
        p_ang = np.array(p_ang)
        
        slope, intercept, *_ = linregress(np.arange(steps + 1), p_ang)
        y = slope * np.arange(steps + 1) + intercept

        std = p_ang.std()
        counter = 0 # track number of intersections with mean

        positive = True if p_ang[1] > p_ang[0] else False

        for i in range(1, steps):
            if positive and p_ang[i] <= (y[i] + std_f * std):
                positive = not positive
                counter += 1
            elif not positive and p_ang[i] >= (y[i] - std_f * std):
                positive = not positive
                counter += 1
            
            if counter // 2 >= oscillations:
                print(f"Total steps to reach 2 oscillations: {i}")
                j = i
                break # 2 oscillations
        
        for k in range(j):
            total_loss += loss_hist[k]
        
        return total_loss
    
    return minimize(loss, initial_p, method='Nelder-Mead', constraints=(), callback=print)


def t2_3_opt_p_with_period_no_print(initial_state, initial_p, std_f=1, oscillations=2, steps=10):
    def loss(p):
        p_ang = []
        total_loss = 0
        cp = CartPole(False)
        cp.setState(initial_state)
        loss_hist = []
        action = np.dot(p, initial_state)
        p_ang.append(initial_state[2])
        j = steps

        for _ in range(steps):
            cp.performAction(action)
            state = cp.getState()
            action = np.dot(p, state)
            loss_hist.append(cp.loss())
            p_ang.append(initial_state[2])

        # Count the current number of periods
        p_ang = np.array(p_ang)
        
        slope, intercept, *_ = linregress(np.arange(steps + 1), p_ang)
        y = slope * np.arange(steps + 1) + intercept

        std = p_ang.std()
        counter = 0 # track number of intersections with mean

        positive = True if p_ang[1] > p_ang[0] else False

        for i in range(1, steps):
            if positive and p_ang[i] <= (y[i] + std_f * std):
                positive = not positive
                counter += 1
            elif not positive and p_ang[i] >= (y[i] - std_f * std):
                positive = not positive
                counter += 1
            
            if counter // 2 >= oscillations:
                j = i
                break # 2 oscillations
        
        for k in range(j):
            total_loss += loss_hist[k]
        
        return total_loss
    
    return minimize(loss, initial_p, method='Nelder-Mead', constraints=(), callback=None)

from NonLinearObserver_ import NonLinearObserver

def t2_1_vary_sigma():
    n_basis = (2 ** 5) * 10
    def simulate(X):
        cp = CartPole(False)
        cp.setState(X)
        cp.performAction()
        return cp.getState() - X # Change in state

    seed = 10

    sampler = qmc.Sobol(d=4, seed=seed)
    X = sampler.random_base2(m=10)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    XI = X[:n_basis]
    SIGMA = X.std(axis=0)

    residuals_hist = np.zeros((4, 9, 4))
    sigma_hist = np.zeros((4, 9))

    Y = np.apply_along_axis(simulate, axis=1, arr=X)

    for i in range(4):
        a = SIGMA[i] / 3
        xrange = np.linspace(SIGMA[i] - a, SIGMA[i] + a, 9)
        print(SIGMA)
        print(xrange)
        for j, f in enumerate(xrange):
            s = np.copy(SIGMA)
            s[i] = f
            K_NM = K(X, XI, s)
            _, residuals, _, _ = np.linalg.lstsq(K_NM, Y)
            residuals_hist[i][j] = residuals
            sigma_hist[i][j] = f

    return sigma_hist, residuals_hist, SIGMA

from shared import STATE_LABELS

def t2_1_plot_sigma(x, y, sigma, fig, axs):
    axs[0,0].plot(x[0, :], y[0, :], label=STATE_LABELS[:4], linestyle='solid')
    axs[0,1].plot(x[1, :], y[1, :], linestyle='solid')
    axs[1,0].plot(x[2, :], y[2, :], linestyle='solid')
    axs[1,1].plot(x[3, :], y[3, :], linestyle='solid')

    #Set titles
    ylabels = ["Residuals"] * 4
    xlabels = [r"$\sigma_{}$".format(i) for i in range(4)]

    axs[0,0].set(xlabel=xlabels[0])
    axs[0,0].set(ylabel=ylabels[0])
    axs[0,1].set(xlabel=xlabels[1])
    axs[0,1].set(ylabel=ylabels[1])
    axs[1,0].set(xlabel=xlabels[2])
    axs[1,0].set(ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3])
    axs[1, 1].set(ylabel=ylabels[3])

    # axs[0,0].set_yscale('log')
    # axs[0,1].set_yscale('log')
    # axs[1,0].set_yscale('log')
    # axs[1,1].set_yscale('log')

    # print(x[3, 0], sigma[0])
    # axs[0,0].scatter(x[3, 0], y[3, 0], color='red', s=72)
    # axs[0,1].scatter(x[3, 1], y[3, 1], color='red', s=72)
    # axs[1,0].scatter(x[3, 2], y[3, 2], color='red', s=72)
    # axs[1,1].scatter(x[3, 3], y[3, 3], color='red', s=72)
    axs[0,0].axvline(x = sigma[0], color = 'k')
    axs[0,1].axvline(x = sigma[1], color = 'k')
    axs[1,0].axvline(x = sigma[2], color = 'k')
    axs[1,1].axvline(x = sigma[3], color = 'k')


def t2_1_plot_states_contour(x, y, z, fig, axs, colors=None, xlabel="", ylabel="", xlim=(-10, 10), ylim=(-15, 15), levels=LEVELS):
    def format(i, z):
        n = x.shape[0]
        z = z[1:, i]
        return np.reshape(z, newshape=(n, n))
    
    z0 = format(0, z)
    z1 = format(1, z)
    z2 = format(2, z)
    z3 = format(3, z)

    c1 = axs[0].contour(x, y, z0, colors=colors, levels=5)
    c2 = axs[1].contour(x, y, z1, colors=colors, levels=5)
    c3 = axs[2].contour(x, y, z2, colors=colors, levels=5)
    c4 = axs[3].contour(x, y, z3, colors=colors, levels=5)

    cntr1 = axs[0].contourf(x, y, z0, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr2 = axs[1].contourf(x, y, z1, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr3 = axs[2].contourf(x, y, z2, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr4 = axs[3].contourf(x, y, z3, linestyles='solid', negative_linestyles='dashed', levels=levels)

    if colors != 'white':
        fig.colorbar(cntr1, ax=axs[0])
        fig.colorbar(cntr2, ax=axs[1])
        fig.colorbar(cntr3, ax=axs[2])
        fig.colorbar(cntr4, ax=axs[3])

    # axs[0, 0].plot(x, y, 'ko', ms=3)
    # axs[0, 1].plot(x, y, 'ko', ms=3)
    # axs[1, 0].plot(x, y, 'ko', ms=3)
    # axs[1, 1].plot(x, y, 'ko', ms=3)

    axs[0].set(xlim=xlim, ylim=ylim)
    axs[1].set(xlim=xlim, ylim=ylim)
    axs[2].set(xlim=xlim, ylim=ylim)
    axs[3].set(xlim=xlim, ylim=ylim)

    axs[0].set_title(DELTA + STATE0)
    axs[1].set_title(DELTA + STATE1)
    axs[2].set_title(DELTA + STATE2)
    axs[3].set_title(DELTA + STATE3)

    axs[0].set(xlabel=xlabel)
    axs[0].set(ylabel=ylabel)
    axs[1].set(xlabel=xlabel)
    axs[1].set(ylabel=ylabel)
    axs[2].set(xlabel=xlabel)
    axs[2].set(ylabel=ylabel)
    axs[3].set(xlabel=xlabel)
    axs[3].set(ylabel=ylabel)

    axs[0].clabel(c1, c1.levels, inline=True, colors=colors)
    axs[1].clabel(c2, c2.levels, inline=True, colors=colors)
    axs[2].clabel(c3, c3.levels, inline=True, colors=colors)
    axs[3].clabel(c4, c4.levels, inline=True, colors=colors)


def t2_3_plot_optimisation_contour(x, y, z, fig, ax, colors=None, xlabel="", ylabel="", xlim=(-10, 10), ylim=(-15, 15), levels=LEVELS):    
    z0 = np.reshape(z, newshape=(x.shape[0], y.shape[0]))

    c1 = ax.contour(x, y, z0, colors=colors, levels=5)

    cntr1 = ax.contourf(x, y, z0, linestyles='solid', negative_linestyles='dashed', levels=levels)
    fig.colorbar(cntr1, ax=ax)
    ax.set(xlim=xlim, ylim=ylim)
    # ax.set_title("Controllable space")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.clabel(c1, c1.levels, inline=True, colors=colors)


def t2_3_roll_out(x, y, y_hat, y_no_action, fig, axs, color=None, label="", xlabel="", ylabel="", xlim=None, ylim=None):
    axs[0,0].plot(x, y[:, 0], color=color, linestyle='solid', label='actual')
    axs[0,1].plot(x, y[:, 1], color=color, linestyle='solid')
    axs[1,0].plot(x, y[:, 2], color=color, linestyle='solid')
    axs[1,1].plot(x, y[:, 3], color=color, linestyle='solid')

    axs[0,0].plot(x, y_hat[:, 0], color=color, linestyle='dashed', label='predicted')
    axs[0,1].plot(x, y_hat[:, 1], color=color, linestyle='dashed')
    axs[1,0].plot(x, y_hat[:, 2], color=color, linestyle='dashed')
    axs[1,1].plot(x, y_hat[:, 3], color=color, linestyle='dashed')

    axs[0,0].plot(x, y_no_action[:, 0], color=color, linestyle='dotted', label='no action')
    axs[0,1].plot(x, y_no_action[:, 1], color=color, linestyle='dotted')
    axs[1,0].plot(x, y_no_action[:, 2], color=color, linestyle='dotted')
    axs[1,1].plot(x, y_no_action[:, 3], color=color, linestyle='dotted')

    #Set titles
    ylabels = [ylabel] * 4 if ylabel else [STATE0, STATE1, STATE2, STATE3]
    xlabels = [xlabel] * 4 if xlabel else [STATE0, STATE1, STATE2, STATE3]

    axs[0,0].set(xlabel=xlabels[0])
    axs[0,0].set(ylabel=ylabels[0])
    axs[0,1].set(xlabel=xlabels[1])
    axs[0,1].set(ylabel=ylabels[1])
    axs[1,0].set(xlabel=xlabels[2])
    axs[1,0].set(ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3])
    axs[1, 1].set(ylabel=ylabels[3])

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0, 0].set(xlim=xlim, ylim=ylim)
    axs[0, 1].set(xlim=xlim, ylim=ylim)
    axs[1, 0].set(xlim=xlim, ylim=ylim)
    axs[1, 1].set(xlim=xlim, ylim=ylim)




    







