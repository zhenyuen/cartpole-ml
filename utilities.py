from CartPole import *

POS_LOW = -10
POS_HIGH = 10
VEL_LOW = -10
VEL_HIGH = 10
ANG_VEL_LOW = -15
ANG_VEL_HIGH = 15
ANG_LOW = -np.pi
ANG_HIGH = np.pi
STATE0 = r"cart position, $x\hspace{2mm}(m)$"
STATE1 = r"cart velocity, $\dot{x}\hspace{2mm}(m/s)$"
STATE2 = r"pole angle, $\theta\hspace{2mm}(rad)$"
STATE3 = r"pole velocity, $\dot{\theta}\hspace{2mm}(rad/s)$"


STABLE_POS = 0
STABLE_VEL = 0
STABLE_ANG = np.pi
STABLE_ANG_VEL = 0

UNSTABLE_POS = 0
UNSTABLE_VEL = 0
UNSTABLE_ANG = 0
UNSTABLE_ANG_VEL = 0

DELTA = r"$\Delta$ "
LEVELS = 20
STEPS = 5 # In multiples of 50

def simulate(state, steps=50, visual=False, remap_angle=False):
    cp = CartPole(visual)
    cp.setState(state)
    cp.sim_steps = 1
    cp.delta_time = 0.1
    steps = np.arange(0, steps + 1)
    states = cp.getState()

    for _ in steps[:-1]:
        cp.performAction()
        states = np.vstack([states, cp.getState()])
    
    steps = np.tile(steps, (4, 1))
    return steps.T, states


def get_subplot(title):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    fig.suptitle(title)
    return fig, axs



def plot_states(x, y, fig, axs, color=None, label="", xlabel="", ylabel="", linestyle=None):
    axs[0,0].plot(x, y[:, 0], color=color, label=label, linestyle=linestyle)
    axs[0,1].plot(x, y[:, 1], color=color, linestyle=linestyle)
    axs[1,0].plot(x, y[:, 2], color=color, linestyle=linestyle)
    axs[1,1].plot(x, y[:, 3], color=color, linestyle=linestyle)

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

    # axs[0, 0].set(xlim=xlim, ylim=ylim)
    # axs[0, 1].set(xlim=xlim, ylim=ylim)
    # axs[1, 0].set(xlim=xlim, ylim=ylim)
    # axs[1, 1].set(xlim=xlim, ylim=ylim)

    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

def plot_all_states_in_a_subplot(x, y, fig, ax, color=None, labels="", xlabel="", ylabel="", linestyle="solid"):
    ax.plot(x, y[:, 0], color=color, linestyle=linestyle)
    ax.plot(x, y[:, 1], color=color, linestyle=linestyle)
    ax.plot(x, y[:, 2], color=color, linestyle=linestyle)
    ax.plot(x, y[:, 3], color=color, linestyle=linestyle)

    #Set titles
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)

    ax.grid()
    

def plot_states_contour(x, y, z, fig, axs, colors=None, xlabel="", ylabel="", xlim=(-10, 10), ylim=(-15, 15), levels=LEVELS):
    def format(i, z):
        n = x.shape[0]
        z = z[1:, i]
        return np.reshape(z, newshape=(n, n))
    
    z0 = format(0, z)
    z1 = format(1, z)
    z2 = format(2, z)
    z3 = format(3, z)

    c1 = axs[0,0].contour(x, y, z0, colors=colors, levels=5)
    c2 = axs[0,1].contour(x, y, z1, colors=colors, levels=5)
    c3 = axs[1,0].contour(x, y, z2, colors=colors, levels=5)
    c4 = axs[1,1].contour(x, y, z3, colors=colors, levels=5)

    cntr1 = axs[0, 0].contourf(x, y, z0, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr2 = axs[0, 1].contourf(x, y, z1, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr3 = axs[1, 0].contourf(x, y, z2, linestyles='solid', negative_linestyles='dashed', levels=levels)
    cntr4 = axs[1, 1].contourf(x, y, z3, linestyles='solid', negative_linestyles='dashed', levels=levels)

    if colors != 'white':
        fig.colorbar(cntr1, ax=axs[0, 0])
        fig.colorbar(cntr2, ax=axs[0, 1])
        fig.colorbar(cntr3, ax=axs[1, 0])
        fig.colorbar(cntr4, ax=axs[1, 1])

    # axs[0, 0].plot(x, y, 'ko', ms=3)
    # axs[0, 1].plot(x, y, 'ko', ms=3)
    # axs[1, 0].plot(x, y, 'ko', ms=3)
    # axs[1, 1].plot(x, y, 'ko', ms=3)

    axs[0, 0].set(xlim=xlim, ylim=ylim)
    axs[0, 1].set(xlim=xlim, ylim=ylim)
    axs[1, 0].set(xlim=xlim, ylim=ylim)
    axs[1, 1].set(xlim=xlim, ylim=ylim)

    axs[0,0].set_title(DELTA + STATE0)
    axs[0,1].set_title(DELTA + STATE1)
    axs[1,0].set_title(DELTA + STATE2)
    axs[1,1].set_title(DELTA + STATE3)

    axs[0,0].set(xlabel=xlabel)
    axs[0,0].set(ylabel=ylabel)
    axs[0,1].set(xlabel=xlabel)
    axs[0,1].set(ylabel=ylabel)
    axs[1,0].set(xlabel=xlabel)
    axs[1,0].set(ylabel=ylabel)
    axs[1, 1].set(xlabel=xlabel)
    axs[1, 1].set(ylabel=ylabel)

    axs[0,0].clabel(c1, c1.levels, inline=True, colors=colors)
    axs[0,1].clabel(c2, c2.levels, inline=True, colors=colors)
    axs[1,0].clabel(c3, c3.levels, inline=True, colors=colors)
    axs[1,1].clabel(c4, c4.levels, inline=True, colors=colors)



def plot_scatter(x, y, fig, axs, color=None, label="", xlabels="", ylabels="", diag=True):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], y[:, 0], color=color, label=label)
    axs[0,1].scatter(x[:, 1], y[:, 1], color=color)
    axs[1,0].scatter(x[:, 2], y[:, 2], color=color)
    axs[1,1].scatter(x[:, 3], y[:, 3], color=color)

    #Set titles
    ylabels = ylabels if ylabels else [STATE0, STATE1, STATE2, STATE3]
    xlabels = xlabels * 4 if xlabels else [STATE0, STATE1, STATE2, STATE3]

    axs[0,0].set(xlabel=xlabels[0])
    axs[0,0].set(ylabel=ylabels[0])
    axs[0,1].set(xlabel=xlabels[1])
    axs[0,1].set(ylabel=ylabels[1])
    axs[1,0].set(xlabel=xlabels[2])
    axs[1,0].set(ylabel=ylabels[2])
    axs[1, 1].set(xlabel=xlabels[3])
    axs[1, 1].set(ylabel=ylabels[3])

    if diag:
        axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
        axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')
        

def plot_all_states_in_a_subplot_1_3(x, y, fig, ax, colors=None, labels="", xlabel="", ylabel="", linestyle='solid'):
    ax.plot(x, y[:, 0], color=colors[0], linestyle=linestyle)
    ax.plot(x, y[:, 1], color=colors[1], linestyle=linestyle)
    ax.plot(x, y[:, 2], color=colors[2], linestyle=linestyle)
    ax.plot(x, y[:, 3], color=colors[3], linestyle=linestyle)

    #Set titles
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)


def run_simulation(X, steps, remap=True):
    cp = CartPole(False)
    state = X.copy()
    cp.setState(state)
    for _ in steps:
        cp.performAction()
        t = cp.getState()
        t[2] = remap_angle(t[2])
        state = np.vstack([state, t])
    return state

def run_linear_model(X, W, steps, remap=True):
    state = X.copy()
    for _ in steps:
        X += (X @ W)
        # X = X.at[2].set(remap_angle(X[2]))
        X[2] = remap_angle(X[2])
        state = np.vstack([state, X])
    return state
