"""
fork from python-rl and pybrain for visualization
"""
#import numpy as np
import numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt
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
    UNSTABLE_EQ,
    STABLE_EQ,
    SEED,
)

# SIG = np.array([0.05, 0.05, 1, 0.5])

# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
     
def remap_angle(theta):
    return _remap_angle(theta)
    
def _remap_angle(theta, tolerance=0.02):
    # while theta < - (np.pi + tolerance):
    #     theta += 2. * np.pi
    # while theta > (np.pi + tolerance):
    #     theta -= 2. * np.pi
    # return theta
    return _custom_remap(theta, tolerance)
    

## loss function given a state vector. the elements of the state vector are
## [cart location, cart velocity, pole angle, pole angular velocity]
def _loss(state, target=0, sig=0.5):
    # sig = np.array([1, 1, 1, 1])
    diff = (state - target) / sig
    return 1-np.exp(-np.dot(diff, diff) * 0.5)

def loss(state, target=0, sig=0.5):
    return _loss(state, target, sig)

class CartPole:
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """

    def __init__(self, visual=False):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi    # angle is defined to be zero when the pole is upright, pi when hanging vertically down
        self.pole_velocity = 0.0
        self.visual = visual

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        self.pole_length = 0.5 
        self.pole_mass = 0.5 

        self.mu_c = 0.001 #   # friction coefficient of the cart
        self.mu_p = 0.001 # # friction coefficient of the pole
        self.sim_steps = 50       # number of Euler integration steps to perform in one go
        self.delta_time = 0.2        # time step of the Euler integrator
        self.max_force = 20.
        self.gravity = 9.8
        self.cart_mass = 0.5

        # for plotting
        self.cartwidth = 1.0
        self.cartheight = 0.2

        if self.visual:
            self.drawPlot()

    def setState(self, state):
        self.cart_location = state[0]
        self.cart_velocity = state[1]
        self.pole_angle = state[2]
        self.pole_velocity = state[3]
            
    def getState(self):
        return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity])

    # reset the state vector to the initial state (down-hanging pole)
    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = np.pi
        self.pole_velocity = 0.0

    # This is where the equations of motion are implemented
    def performAction(self, action = 0.0):
        # prevent the force from being too large
        force = self.max_force * np.tanh(action/self.max_force)

        # integrate forward the equations of motion using the Euler method
        for step in range(self.sim_steps):
            s = np.sin(self.pole_angle)
            c = np.cos(self.pole_angle)
            m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
            
            cart_accel = (2.0*(self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+2.0*(force-self.mu_c*self.cart_velocity))\
                -3.0*self.pole_mass*self.gravity*c*s + 6.0*self.mu_p*self.pole_velocity*c/self.pole_length)/m
            
            pole_accel = (-3.0*c*(2.0/self.pole_length)*(self.pole_length/2.0*self.pole_mass*(self.pole_velocity**2)*s + force-self.mu_c*self.cart_velocity)+\
                6.0*(self.cart_mass+self.pole_mass)/(self.pole_mass*self.pole_length)*\
                (self.pole_mass*self.gravity*s - 2.0/self.pole_length*self.mu_p*self.pole_velocity) \
                )/m
            # Update state variables
            dt = (self.delta_time / float(self.sim_steps))
            # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
            self.cart_velocity += dt * cart_accel
            self.pole_velocity += dt * pole_accel
            self.pole_angle    += dt * self.pole_velocity
            self.cart_location += dt * self.cart_velocity
        
        if self.visual:
            self._render()

    # remapping as a member function
    def remap_angle(self):
        self.pole_angle = _remap_angle(self.pole_angle)
    
    # the loss function that the policy will try to optimise (lower) as a member function
    def loss(self):
        return _loss(self.getState())
    

   # the following are graphics routines
    def drawPlot(self):
        ion()
        self.fig = plt.figure()
        # draw cart
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.box = Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight), 
                             width=self.cartwidth, height=self.cartheight)
        self.axes.add_artist(self.box)
        self.box.set_clip_box(self.axes.bbox)

        # draw pole
        self.pole = Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle)], 
                           [0, np.cos(self.pole_angle)], linewidth=3, color='black')
        self.axes.add_artist(self.pole)
        self.pole.set_clip_box(self.axes.bbox)

        # set axes limits
        # self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-0.5, 2)
        


    def _render(self):
        self.box.set_x(self.cart_location - self.cartwidth / 2.0)
        self.pole.set_xdata([self.cart_location, self.cart_location + np.sin(self.pole_angle)])
        self.pole.set_ydata([0, np.cos(self.pole_angle)])
        self.fig.show()
        
        plt.pause(0.2)

    def _get_state(self):
        return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity])

    def simulate_actual(self, state, remap, n_steps=None, time=0.0):
        state, action = state[:4], state[4]
        if n_steps is None: n_steps = int(np.ceil(time / self.delta_time))
        
        state_hist = np.zeros((n_steps + 1, 4))
        time_hist = np.zeros(n_steps + 1)
        action_hist = np.zeros(n_steps)
        action_hist[0] = action

        self.setState(state)
        state_hist[0] = state
        for i in np.arange(1, n_steps + 1):
            time_hist[i] = i * self.delta_time
            self.performAction(action_hist[i - 1])
            if remap: self.remap_angle()
            state = self._get_state()
            state_hist[i] = state
        self.reset()
        return time_hist, state_hist
    
    def simulate(self, state, remap, n_steps=None, time=0.0):
        state, action = state[:4], state[4]
        if n_steps is None: n_steps = int(np.ceil(time / self.delta_time))
        
        state_hist = np.zeros((n_steps + 1, 4))
        time_hist = np.zeros(n_steps + 1)
        action_hist = np.zeros(n_steps)
        action_hist[0] = action

        self.setState(state)
        state_hist[0] = state
        for i in np.arange(1, n_steps + 1):
            time_hist[i] = i * self.delta_time
            self.performAction(action_hist[i - 1])
            if remap: self.remap_angle()
            state = self.getState()
            state_hist[i] = state

        self.reset()
        return time_hist, state_hist
    
    def simulate_with_action(self, state, remap, action_seq=0, n_steps=None, time=0.0):
        state, _ = state[:4], state[4]
        if n_steps is None: n_steps = int(np.ceil(time / self.delta_time))

        state_hist = np.zeros((n_steps + 1, 4))
        time_hist = np.zeros(n_steps + 1)

        padded_action_seq = np.pad(action_seq, (0, n_steps - len(action_seq)), mode='constant')

        self.setState(state)
        state_hist[0] = state
        for i in np.arange(1, n_steps + 1):
            time_hist[i] = i * self.delta_time
            self.performAction(padded_action_seq[i - 1])
            if remap: self.remap_angle()
            state = self.getState()
            state_hist[i] = state

        self.reset()
        return time_hist, state_hist
    

    def simulate_with_feedback(self, state, remap, controller=None, n_steps=None, time=0.0):
        state, _ = state[:4], state[4]
        if n_steps is None: n_steps = int(np.ceil(time / self.delta_time))
        if controller is None: controller = BaseController()
        
        state_hist = np.zeros((n_steps + 1, 4))
        time_hist = np.zeros(n_steps + 1)

        self.setState(state)
        state_hist[0] = state
        for i in np.arange(1, n_steps + 1):
            time_hist[i] = i * self.delta_time
            feedback_action = controller.feedback_action(state)
            self.performAction(feedback_action)
            if remap: self.remap_angle()
            state = self.getState()
            state_hist[i] = state

        self.reset()
        return time_hist, state_hist
    
    def total_loss(self, state, remap, controller=None, n_steps=None, time=0.0):
        _, state_hist = self.simulate_with_feedback(state, remap, controller, n_steps, time)
        total_loss = 0

        for _ in state_hist:
            total_loss += loss()
        return total_loss
    
    def total_loss_2(self, state, remap, controller, n_steps=None, time=0.0, min_loss=float('inf')):
        state, _ = state[:4], state[4]
        if n_steps is None: n_steps = int(np.ceil(time / self.delta_time))
        
        total_loss = 0
        self.setState(state)

        for i in np.arange(n_steps):
            feedback_action = controller.feedback_action(state)
            self.performAction(feedback_action)
            if remap: self.remap_angle()
            total_loss += self.loss()
            state = self.getState()

        self.reset()
        return total_loss
    

    # # Does not work
    # def perform_action_numba(self, action = 0.0):
    #     # prevent the force from being too large
    #     start = self.getState()
    #     max_force=self.max_force
    #     sim_steps=self.sim_steps
    #     state=start
    #     cart_mass=self.cart_mass
    #     pole_mass=self.pole_mass
    #     pole_length=self.pole_length
    #     mu_c=self.mu_c
    #     gravity=self.gravity
    #     mu_p=self.mu_p
    #     delta_time=self.delta_time

    #     state = perform_action(action=action,
    #         max_force=max_force,
    #         sim_steps=sim_steps,
    #         state=start,
    #         cart_mass=cart_mass,
    #         pole_mass=pole_mass,
    #         pole_length=pole_length,
    #         mu_c=mu_c,
    #         gravity=gravity,
    #         mu_p=mu_p,
    #         delta_time=delta_time
    #     )
    #     return np.array(state)
    
    # # Does not work
    # def simulate_numba(self, time, state, remap, controller=None):
    #     n_steps = int(np.ceil(time / self.delta_time))
    #     if controller is None: controller = BaseController()
        
    #     state_hist = np.zeros((n_steps + 1, 4))
    #     time_hist = np.zeros(n_steps + 1)

    #     self.setState(state)
    #     for i in np.arange(1, n_steps + 1):
    #         time_hist[i] = i * self.delta_time
    #         feedback_action = controller.feedback_action(state)
    #         self.perform_action_numba(action=feedback_action)
    #         if remap: self.remap_angle()
    #         state = self.getState()
    #         state_hist[i] = state

    #     self.reset()
    #     return time_hist, state_hist
    
        

class BaseController():
    def feedback_action(self, state):
        return 0.0
    

# Does not work
# @njit
# def perform_action(action, max_force, sim_steps, state, cart_mass, pole_mass, pole_length, mu_c, gravity, mu_p, delta_time):
#     cart_location, cart_velocity, pole_angle, pole_velocity = state
    

#      # prevent the force from being too large
#     force = max_force * np.tanh(action/max_force)
#     # force = max_force
#     # integrate forward the equations of motion using the Euler method
#     for step in range(sim_steps):
#         s = np.sin(pole_angle)
#         c = np.cos(pole_angle)
#         m = 4.0*(cart_mass+pole_mass)-3.0*pole_mass*(c**2)
        
#         cart_accel = (2.0*(pole_length*pole_mass*(pole_velocity**2)*s+2.0*(force-mu_c*cart_velocity))\
#             -3.0*pole_mass*gravity*c*s + 6.0*mu_p*pole_velocity*c/pole_length)/m
        
#         pole_accel = (-3.0*c*(2.0/pole_length)*(pole_length/2.0*pole_mass*(pole_velocity**2)*s + force-mu_c*cart_velocity)+\
#             6.0*(cart_mass+pole_mass)/(pole_mass*pole_length)*\
#             (pole_mass*gravity*s - 2.0/pole_length*mu_p*pole_velocity) \
#             )/m

#         # Update state variables
#         dt = (delta_time / float(sim_steps))
#         # Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
#         cart_velocity += (dt * cart_accel),
#         pole_velocity += (dt * pole_accel),
#         pole_angle    += (dt * pole_velocity),
#         cart_location += (dt * cart_velocity)
    
#     return cart_location, cart_velocity, pole_angle, pole_velocity
        
def _custom_remap(theta, tolerance=0.02):
    if theta < - (np.pi + tolerance):
        imm = theta % (-2*np.pi)
        if imm < -np.pi: return imm + (2 * np.pi)
        return imm
    elif theta > (np.pi + tolerance):
        imm = theta % (2*np.pi)
        if imm > np.pi: return imm - (2 * np.pi)
        return imm
    return theta


if __name__ == "__main__":
    test = np.random.uniform(-3 *np.pi, 3 * np.pi, size=1000)
    for t in test:
        if _custom_remap(t) != _remap_angle(t):
            print(t)
            print(_custom_remap(t))
            print(_remap_angle(t))
            assert False
