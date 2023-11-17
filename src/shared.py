
import numpy as np
import matplotlib as mpl
from scipy.stats import qmc, linregress

STABLE_POS = 0.0
STABLE_VEL = 0.0
STABLE_ANG = np.pi
STABLE_ANG_VEL = 0.0
STABLE_EQ = np.array([STABLE_POS, STABLE_VEL, STABLE_ANG, STABLE_ANG_VEL])

UNSTABLE_POS = 0.0
UNSTABLE_VEL = 0.0
UNSTABLE_ANG = 0.0
UNSTABLE_ANG_VEL = 0.0
UNSTABLE_EQ = np.array([UNSTABLE_POS, UNSTABLE_VEL, UNSTABLE_ANG, UNSTABLE_ANG_VEL], dtype='float64')

POS_LOW = -10.0
POS_HIGH = 10.0
VEL_LOW = -10.0
VEL_HIGH = 10.0
ANG_VEL_LOW = -15.0
ANG_VEL_HIGH = 15.0
ANG_LOW = -np.pi
ANG_HIGH = np.pi
FORCE_LOW = -15
FORCE_HIGH = 15

STATE0 = r"cart position, $x\ \ \ (m)$"
STATE1 = r"cart velocity, $\dot x\ \ \ (m/s)$"
STATE2 = r"pole angle, $\theta\ \ \ (rad)$"
STATE3 = r"pole velocity, $\dot \theta\ \ \ (rad/s)$"
STATE4 = r"action force, $f\ \ \ (N)$"
STATE_LABELS = [STATE0, STATE1, STATE2, STATE3, STATE4]

SEED = 10

N_CONTOUR_LEVELS = 20
DELTA_SYM = r"$\Delta$ "


def np_to_string(x):
    return np.array2string(x, precision=2, separator=', ', suppress_small=True, floatmode='fixed')

def get_initial_states(action=0.0):
    points = {
        "stable+tinyosc": (0.0, 0.01, np.pi, 0.01, action),
        "stable+smallosc": (0.0, 2.0, np.pi+0.1, 2.0, action),
        "stable+largeosc": (0.0, 8.0, np.pi+0.1, 8.0, action),
        "stable+completeosc": (0.0, 14.0, np.pi+0.1, 14.0, action),
        "unstable+tinyperturb": (0.0, 0.01, 0.01, 0.01, action),
        "unstable": (0.0, 0.01, 0.5, 0.01, action),
        "unstable+smallvel": (0.0, 2.0, 0.5, 2.0, action),
        "unstable+largevel": (0.0, 8.0, 0.5, 8.0, action),
    }

    for p in points:
        points[p] = np.array(points[p])
    #     return (
    #     (0, 0.01, np.pi, 0.01), # stable equilibrium, simple oscillations
    #     (0, 2, np.pi+0.1, 2), # stable equilibrium
    #     (0, 2, np.pi+0.1, 14), # stable equilibrium, large pole velocity
    #     (0, 14, np.pi+0.1, 2), # Unstable equilibrium, large pole velocity
    #     (0, 14, np.pi+0.1, 14), # Unstable equilibrium, large pole and cart velocity
    #     (0, 8, np.pi+0.1, 8), # Unstable equilibrium, large pole and cart velocity, but no complete oscillation
    #     (0, 2, np.pi / 2, 2), # Halfway point,
    #     (0, 0.01, 0.1, 0.01), # Unstable equilibrium, no initial velocities
    #     (0, 2, 0.1, 2), # Unstable equilibrium
    #     (0, 2, 0.1, 2), # Unstable equilibrium
    #     (0, 2, 0.1, 14), # Unstable equilibrium, large pole velocity
    #     (0, 14, 0.1, 2), # Unstable equilibrium, large pole velocity
    #     (0, 14, 0.1, 14), # Unstable equilibrium, large pole and cart velocity
    #     (0, 8, 0.1, 8), # Unstable equilibrium, large pole and cart velocity, but no complete oscillation
    #     (0, 8, 0.1, 8) # Halfway point, large pole and cart velocity
    # )
    return points


def get_sobol_points(m, d):
    sampler = qmc.Sobol(
            d=d,
            seed=SEED,
        )
    
    X = sampler.random_base2(m)
    X[:, 0] = X[:, 0] * (POS_HIGH - POS_LOW) + POS_LOW
    X[:, 1] = X[:, 1] * (VEL_HIGH - VEL_LOW) + VEL_LOW
    X[:, 2] = X[:, 2] * (ANG_HIGH - ANG_LOW) + ANG_LOW
    X[:, 3] = X[:, 3] * (ANG_VEL_HIGH - ANG_VEL_LOW) + ANG_VEL_LOW
    if d == 5: X[:, 4] = X[:, 4] * 0
    return X

def plot_rollout(x, y, axs, fig, color=None, label=None, linestyle='solid', linewidth=None):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle=linestyle, linewidth=linewidth)
    axs[0,1].plot(x, y[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
    axs[1,0].plot(x, y[:, 2], color=color, linestyle=linestyle, linewidth=linewidth)
    axs[1,1].plot(x, y[:, 3], color=color, linestyle=linestyle, linewidth=linewidth)

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

def plot_rollout_merge(x, y, axs, fig, color=None, label=None, linestyle='solid'):
    axs[0].plot(x, y[:, 0], color=color, label=label, linestyle=linestyle)
    axs[1].plot(x, y[:, 1], color=color, linestyle=linestyle)
    axs[2].plot(x, y[:, 2], color=color, linestyle=linestyle)
    axs[3].plot(x, y[:, 3], color=color, linestyle=linestyle)

    #Set titles
    ylabels = STATE_LABELS
    xlabels = [r"time $(s)$"] * 4

    axs[0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    axs[1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    axs[2].set(xlabel=xlabels[2], ylabel=ylabels[2])
    axs[3].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()


def plot_rollout_single(x, y, ax, fig, color=None, label=None, linestyle='solid', legend=True):
    labels = STATE_LABELS
    
    if legend:
        ax.plot(x, y[:, 0], color=color[0], label=labels[0], linestyle=linestyle)
        ax.plot(x, y[:, 1], color=color[1], label=labels[1], linestyle=linestyle)
        ax.plot(x, y[:, 2], color=color[2], label=labels[2], linestyle=linestyle)
        ax.plot(x, y[:, 3], color=color[3], label=labels[3], linestyle=linestyle)
    else:
        ax.plot(x, y[:, 0], color=color[0], linestyle=linestyle)
        ax.plot(x, y[:, 1], color=color[1], linestyle=linestyle)
        ax.plot(x, y[:, 2], color=color[2], linestyle=linestyle)
        ax.plot(x, y[:, 3], color=color[3], linestyle=linestyle)

    #Set titles
    ylabel = "Response"
    xlabel = r"time $(s)$"

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid()




def plot_loss_1D(x, y, ax, fig, color=None, label=None, linestyle='solid', time=0):
    ax.plot(x, y, color=color, label=label, linestyle=linestyle)
    ax.set(xlabel="policy coefficient", ylabel=f"Min. total loss after {time} seconds")

def plot_loss_2D(x, y, z, p, q, ax, fig, colors=None, xlabel="", ylabel=""):
    c1 = ax.contour(x, y, z, colors=colors)
    cntr1 = ax.contourf(x, y, z, linestyles='solid', cmap='viridis')
    fig.colorbar(cntr1, ax=ax)
    ax.set(xlabel=f"$P_{p}$", ylabel=f"$P_{q}$")
    ax.clabel(c1, c1.levels, inline=True, colors=colors)

def plot_fit_scatter(x, y, axs, fig, color=None, label="", diag=True, s=75):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], y[:, 0], c=color, label=label, s=s)
    axs[0,1].scatter(x[:, 1], y[:, 1], c=color, s=s)
    axs[1,0].scatter(x[:, 2], y[:, 2], c=color, s=s)
    axs[1,1].scatter(x[:, 3], y[:, 3], c=color, s=s)

    #Set titles
    ylabels = [f"predicted {l}" for l in STATE_LABELS]
    xlabels = [f"actual {l}" for l in STATE_LABELS]

    axs[0,0].set(xlabel=xlabels[0])
    axs[0,0].set(ylabel=ylabels[0])
    axs[0,1].set(xlabel=xlabels[1])
    axs[0,1].set(ylabel=ylabels[1])
    axs[1,0].set(xlabel=xlabels[2])
    axs[1,0].set(ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3])
    axs[1, 1].set(ylabel=ylabels[3])

    xlims = zip(x.min(axis=0), x.max(axis=0))
    ylims = zip(y.min(axis=0), y.max(axis=0))

    lims = []
    for i in range(4):
        xlim, ylim = next(xlims), next(ylims)
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        lims.append((lim_min, lim_max))


    if diag:
        axs[0, 0].plot(lims[0], lims[0], ls="--", linewidth=2, color='k')
        axs[0, 1].plot(lims[1], lims[1], ls="--", linewidth=2, color='k')
        axs[1, 0].plot(lims[2], lims[2], ls="--", linewidth=2, color='k')
        axs[1, 1].plot(lims[3], lims[3], ls="--", linewidth=2, color='k')

    
def get_contour_levels(z, z_hat):
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z2 = z[:, :, 2]
    z3 = z[:, :, 3]

    z0_hat = z_hat[:, :, 0]
    z1_hat = z_hat[:, :, 1]
    z2_hat = z_hat[:, :, 2]
    z3_hat = z_hat[:, :, 3]

    z0_lim = (min(z0.min(), z0_hat.min()), max(z0.max(), z0_hat.max()))
    z1_lim = (min(z1.min(), z1_hat.min()), max(z1.max(), z1_hat.max()))
    z2_lim = (min(z2.min(), z2_hat.min()), max(z2.max(), z2_hat.max()))
    z3_lim = (min(z3.min(), z3_hat.min()), max(z3.max(), z3_hat.max()))


    return (
        np.linspace(z0_lim[0], z0_lim[1], N_CONTOUR_LEVELS),
        np.linspace(z1_lim[0], z1_lim[1], N_CONTOUR_LEVELS),
        np.linspace(z2_lim[0], z2_lim[1], N_CONTOUR_LEVELS),
        np.linspace(z3_lim[0], z3_lim[1], N_CONTOUR_LEVELS),
    )


def plot_actual_and_predicted_states_contour(x, y, z, p, q, axs, fig, colors=None, xlabel="", ylabel="", xlim=(-10, 10), ylim=(-15, 15), levels=N_CONTOUR_LEVELS):

    # def format(i, z):
    #     n = x.shape[0]
    #     z = z[1:, i]
    #     return np.reshape(z, newshape=(n, n))
    
    # z0 = format(0, z)
    # z1 = format(1, z)
    # z2 = format(2, z)
    # z3 = format(3, z)

    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z2 = z[:, :, 2]
    z3 = z[:, :, 3]

    c1 = axs[0].contour(x, y, z0, colors=colors, levels=levels[0][::4])
    c2 = axs[1].contour(x, y, z1, colors=colors, levels=levels[1][::4])
    c3 = axs[2].contour(x, y, z2, colors=colors, levels=levels[2][::4])
    c4 = axs[3].contour(x, y, z3, colors=colors, levels=levels[3][::4])



    if colors == 'white':
        cntr1 = axs[0].contourf(x, y, z0, linestyles='solid', levels=levels[0], cmap='viridis')
        cntr2 = axs[1].contourf(x, y, z1, linestyles='solid', levels=levels[1], cmap='viridis')
        cntr3 = axs[2].contourf(x, y, z2, linestyles='solid', levels=levels[2], cmap='viridis')
        cntr4 = axs[3].contourf(x, y, z3, linestyles='solid', levels=levels[3], cmap='viridis')

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

    axs[0].set_title(DELTA_SYM + STATE0)
    axs[1].set_title(DELTA_SYM + STATE1)
    axs[2].set_title(DELTA_SYM + STATE2)
    axs[3].set_title(DELTA_SYM + STATE3)

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

    fig.supxlabel(STATE_LABELS[p])
    fig.supylabel(STATE_LABELS[q])
    


def get_scan_states_ranges(n, limits=(None, None, None, None, None)):
    pos_lim = (POS_LOW, POS_HIGH) if limits[0] is None else limits[0]
    vel_lim = (VEL_LOW, VEL_HIGH) if limits[1] is None else limits[1]
    ang_lim = (ANG_LOW, ANG_HIGH) if limits[2] is None else limits[2]
    ang_vel_lim = (ANG_VEL_LOW, ANG_VEL_HIGH) if limits[3] is None else limits[3]
    force_lim = (FORCE_LOW, FORCE_HIGH) if limits[4] is None else limits[4]

    POS_RANGE = np.linspace(pos_lim[0], pos_lim[1], n)
    VEL_RANGE = np.linspace(vel_lim[0], vel_lim[1], n)
    ANG_RANGE = np.linspace(ang_lim[0], ang_lim[1], n)
    ANG_VEL_RANGE = np.linspace(ang_vel_lim[0], ang_vel_lim[1], n)
    FORCE_RANGE = np.linspace(force_lim[0], force_lim[1], n)
    return POS_RANGE, VEL_RANGE, ANG_RANGE, ANG_VEL_RANGE, FORCE_RANGE


def get_mse(target, model, n_steps=1, m=10, d=5):
    def func(x):
        _, state_hist = target.simulate(state=x, remap=True, n_steps=n_steps)
        return state_hist[1, :] - x[:4]

    def func2(x):
        _, state_hist = model.simulate(state=x, remap=True, n_steps=n_steps)
        return state_hist[1, :] - x[:4]
    
    x = get_sobol_points(m, d)
    actual_change = np.apply_along_axis(
        func,
        axis=1,
        arr=x,
    )
    
    predicted_change = np.apply_along_axis(
        func2,
        axis=1,
        arr=x,
    )

    diff = actual_change - predicted_change
    mse = np.sqrt(np.sum(diff**2)) / actual_change.size
    return mse

def get_actual_and_predicted_change(target, model, n_steps, m=10, d=5):
    def func(x):
        _, state_hist = target.simulate(state=x, remap=False, n_steps=n_steps)
        return state_hist[1, :] - x[:4]

    def func2(x):
        _, state_hist = model.simulate(state=x, remap=False, n_steps=n_steps)
        return state_hist[1, :] - x[:4]
    
    x = get_sobol_points(m, d)
    actual_change = np.apply_along_axis(
        func,
        axis=1,
        arr=x,
    )
    
    predicted_change = np.apply_along_axis(
        func2,
        axis=1,
        arr=x,
    )

    return actual_change, predicted_change


def get_trend(target, initial_state, time, enable_remap=True):
    time_hist, state_hist = target.simulate(state=initial_state, remap=enable_remap, time=time)
    slopes = np.zeros(4)
    intercepts = np.zeros(4)

    for i in range(4):
        slopes[i], intercepts[i], *_ = linregress(time_hist, state_hist[:, i])

    return slopes, intercepts


def get_noise_scales(slopes, intercepts, time_vector, scale=1):
    response = np.zeroes((time_vector.size, 4))
    return response * scale


