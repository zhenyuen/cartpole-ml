import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from scipy.optimize import minimize
from CartPole_ import CartPole
from CartPole_ import loss as cpl
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


def rollout(cp, axs, fig):
    steps = np.arange(40 + 1)
    steps = steps * 0.2
    l = 5
    
    colors = plt.cm.jet(np.linspace(0, 1, l))

    x0 = np.array([0, 0, 0.5, 0])
    cp.setState(x0)

    x = x0[None, ...]
    
    for _ in steps[1: ]:
        # cp.remap_angle()
        x = np.vstack([x, cp.getState()])
        cp.performAction()

    rollout_plotter(steps, x, fig=fig, axs=axs)
    cp.reset()


def rollout_plotter(x, y, fig, axs, color=None, label=""):
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


def optimize_linear_policy_callback(p, counter=count()):
    i = next(counter)
    print(f"Iter {i: >3}: {p}")


def loss(p, cp, x0, n_steps):
    cp.setState(x0)
    total_loss = cp.loss()
    action = np.dot(p, x0)
    for _ in range(n_steps):
        cp.performAction(action)
        
        state = cp.getState()

        if np.any( np.abs(state) > np.array([30, 30, 30, 30]) ) and np.abs(action) > 7:
            return n_steps
    
        total_loss += cp.loss()

        action = np.dot(p, state)
    cp.reset()
    return total_loss


def rollout_loss( initial_state, p, cp):
    state = np.array(initial_state)
    p = np.array(p)
    sim_seconds = 10
    sim_steps = int( sim_seconds / 0.2 )
    loss = 0
    cp.setState(initial_state)
    action = 0
    for i in range( sim_steps ):
        
        # if np.any( np.abs(state) > np.array([10,40,50,1e+3]) ) or np.abs(action) > 1e+3:
        #     return sim_steps
        # print(p, state)
        action = np.dot(p, state)
        loss += cpl( state )
        cp.performAction( action )
        state = cp.getState()
    
    return loss


def optimize_linear_policy(cp, x0, p0, n_steps, callback=optimize_linear_policy_callback):
    func = lambda p: loss(p, cp, x0, n_steps)
    options = {"maxiter": None}
    return minimize(func, p0, method='Nelder-Mead', callback=callback, options=options)


def optimize_linear_policy2(cp, x0, p0, n_steps, callback=optimize_linear_policy_callback):
    func = lambda p: rollout_loss(x0, p, cp)
    options = {"maxiter": None}
    return minimize(func, p0, method='Nelder-Mead', callback=callback, options=options)



def linear_model_linear_policy_rollout(cp, x0, p, fig, axs, n_steps=50):
    x0 = np.array(x0)
    p = np.array(p)

    cp.setState(x0)

    x = x0[None, ...]
    action_hist = []

    steps = np.arange(n_steps + 1) * 0.2

    action = np.dot(p, x0)
    action_hist.append(action)
    for _ in steps[1: ]:
        cp.performAction(action, enable_remap=True)
        # cp.remap_angle()

        state = cp.getState()
        x = np.vstack([x, state])
        
        action = np.dot(p, state)
        action_hist.append(action)

    l = 5
    colors = plt.cm.jet(np.linspace(0, 1, l))

    action_hist = np.array(action_hist)

    linear_model_linear_policy_rollout_plotter(steps, x, action_hist, fig, axs, colors)
    cp.reset()


def linear_model_linear_policy_rollout_plotter(x, y, a, fig, axs, colors):
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

    y_lim = (y.min(), y.max())
    axs[0].set(xlabel=xlabel, ylabel=ylabel)
    axs[0].grid()
    axs[0].set_ylim(y_lim)
    axs[1].set(xlabel=xlabel, ylabel=ylabel)
    axs[1].grid()


    # axs[1].set_yscale('log')


def get_next_search_space(prev_opt, factor=10):
    delta = np.abs(prev_opt) / factor
    limits = []

    for i, p in enumerate(prev_opt):
        limits.append([p - delta[i], p + delta[i]])
    
    return limits


