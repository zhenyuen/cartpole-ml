import numpy as np

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

STATE0 = r"cart position, $x\ \ (m)$"
STATE1 = r"cart velocity, $\dot x\ \ (m/s)$"
STATE2 = r"pole angle, $\theta\ \ (rad)$"
STATE3 = r"pole velocity, $\dot \theta\ \ (rad/s)$"
STATE4 = r"action force, $f\ \ (N)$"
STATE_LABELS = [STATE0, STATE1, STATE2, STATE3, STATE4]



def t4_check_fit_plotter(x, x_hat, axs, fig):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], x_hat[:, 0])
    axs[0,1].scatter(x[:, 1], x_hat[:, 1])
    axs[1,0].scatter(x[:, 2], x_hat[:, 2])
    axs[1,1].scatter(x[:, 3], x_hat[:, 3])

    ylabels = [f"predicted {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0], xlim=(-3, 3), ylim=(-3, 3))
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1], xlim=(-7, 7), ylim=(-7, 7))
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2], xlim=(-7, 7), ylim=(-7, 7))
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3], xlim=(-16, 16), ylim=(-16, 16))

    axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
    axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
    axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
    axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')


def t3_2_check_fit_plotter(x, x_hat, axs, fig):
    axs[0, 0].grid()
    axs[1, 0].grid()
    axs[0, 1].grid()
    axs[1, 1].grid()

    axs[0,0].scatter(x[:, 0], x_hat[:, 0])
    axs[0,1].scatter(x[:, 1], x_hat[:, 1])
    axs[1,0].scatter(x[:, 2], x_hat[:, 2])
    axs[1,1].scatter(x[:, 3], x_hat[:, 3])

    ylabels = [f"predicted {label}" for label in [STATE0, STATE1, STATE2, STATE3]]
    xlabels = [f"actual {label}" for label in [STATE0, STATE1, STATE2, STATE3]]

    axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0], xlim=(1, 8), ylim=(1, 8))
    axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1], xlim=(-4, 12), ylim=(-4, 12))
    axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2], xlim=(-6, 10), ylim=(-6, 10))
    axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3], xlim=(-15, 20), ylim=(-15, 20))

    # axs[0, 0].set(xlabel=xlabels[0], ylabel=ylabels[0])
    # axs[0, 1].set(xlabel=xlabels[1], ylabel=ylabels[1])
    # axs[1, 0].set(xlabel=xlabels[2], ylabel=ylabels[2])
    # axs[1, 1].set(xlabel=xlabels[3], ylabel=ylabels[3])

    axs[0, 0].plot(axs[0, 0].get_xlim(), axs[0, 0].get_ylim(), ls="--", linewidth=2, color='k')
    axs[0, 1].plot(axs[0, 1].get_xlim(), axs[0, 1].get_ylim(), ls="--", linewidth=2, color='k')
    axs[1, 0].plot(axs[1, 0].get_xlim(), axs[1, 0].get_ylim(), ls="--", linewidth=2, color='k')
    axs[1, 1].plot(axs[1, 1].get_xlim(), axs[1, 1].get_ylim(), ls="--", linewidth=2, color='k')