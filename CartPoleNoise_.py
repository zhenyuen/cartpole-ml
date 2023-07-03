from CartPole_ import CartPole, remap_angle
from numpy.random import (
    default_rng,
)
import numpy as np
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

class GaussianNoise():
    # Assume non-correlated and same mean
    def __init__(self, scale, loc) -> None:
        self.rng = default_rng(seed=SEED)
        self.loc = loc 
        self.scale = scale
        self.dim = 4
        self.cov = scale * np.eye(self.dim)
        self.mean = loc * np.ones(self.dim)
    
    def sample(self, size=1):
        noise = self.rng.multivariate_normal(self.mean, self.cov, size)
        return noise[0]
    

class DynamicGaussianNoise():
    # Assume non-correlated and same mean
    def __init__(self, loc, factor, max_scale) -> None:
        self.rng = default_rng(seed=SEED)
        self.loc = loc 
        self.dim = 4
        self.mean = loc * np.ones(self.dim)
        self.factor = factor
        self.max_scale = max_scale
            
    def sample(self, state, size=1):
        cov = (state ** 2) * np.eye(self.dim) * self.factor
        cov_max = self.max_scale * np.eye(self.dim)
        cov = np.minimum(cov, cov_max)
        noise = self.rng.multivariate_normal(self.mean, cov, size)
        return noise[0]

class CartPoleNoisyDyn(CartPole):
    def __init__(self, visual=False):
        super().__init__(visual)

    def performAction(self, action=0):
        super().performAction(action)
        size = len(super().getState())
        noise = self.noise_gen.sample(size)
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle += noise[2]
        self.pole_velocity += noise[3]


class CartPoleNoisyObs(CartPole):
    def __init__(self, visual=False):
        super().__init__(visual)
        self.noise_gen = None

    def getState(self):
        state = super().getState()
        noise = self.noise_gen.sample(1)
        state = state + noise 
        # state[2] = remap_angle(state[2])     # only added when doing 3.2  
        return state


class CartPoleNoisyObsDyn(CartPoleNoisyObs):
    def __init__(self, visual=False):
        super().__init__(visual)

    def performAction(self, action=0, enable_remap=False):
        noise = self.noise_gen.sample()
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle += noise[2]
        self.pole_velocity += noise[3]
        super().performAction(action)



class CartPoleGaussianNoisyObs(CartPoleNoisyObs):
    def __init__(self, visual=False, loc=0, scale=1):
        super().__init__(visual)
        self.noise_gen = GaussianNoise(loc=loc, scale=scale)



class CartPoleGaussianNoisyObs(CartPoleNoisyObs):
    def __init__(self, visual=False, loc=0, scale=1):
        super().__init__(visual)
        self.noise_gen = GaussianNoise(loc=loc, scale=scale)
    

class CartPoleGaussianNoisyObsDyn(CartPoleNoisyObsDyn):
    def __init__(self, visual=False, loc=0, scale=1):
        super().__init__(visual)
        self.noise_gen = GaussianNoise(loc=loc, scale=scale)


class CartPoleGaussianNoisyDyn(CartPoleNoisyDyn):
    def __init__(self, visual=False, loc=0, scale=1):
        super().__init__(visual)
        self.noise_gen = GaussianNoise(loc=loc, scale=scale)



class CartPoleDynamicGaussianNoisyObs(CartPole):
    def __init__(self, visual=False, loc=0, factor=1, max_scale=1):
        super().__init__(visual)
        self.noise_gen = DynamicGaussianNoise(loc=loc, factor=factor, max_scale=max_scale)

    def getState(self):
        state = super().getState()
        noise = self.noise_gen.sample(size=1, state=state)
        state = state + noise 
        return state


class CartPoleDynamicGaussianNoisyObsDyn(CartPole):
    def __init__(self, visual=False, loc=0, factor=1, max_scale=1):
        super().__init__(visual)
        self.noise_gen = DynamicGaussianNoise(loc=loc, factor=factor, max_scale=max_scale)
    
    def getState(self):
        state = super().getState()
        noise = self.noise_gen.sample(size=1, state=state)
        state = state + noise 
        return state

    def performAction(self, action=0):
        actual_state = np.array([self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity])
        noise = self.noise_gen.sample(size=1, state=actual_state)
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle += noise[2]
        self.pole_velocity += noise[3]
        super().performAction(action)


class CartPoleDynamicGaussianNoisyDyn(CartPole):
    def __init__(self, visual=False, loc=0, factor=1, max_scale=1):
        super().__init__(visual)
        self.noise_gen = DynamicGaussianNoise(loc=loc, factor=factor, max_scale=max_scale)

    def performAction(self, action=0):
        actual_state = np.array([self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity])
        noise = self.noise_gen.sample(size=1, state=actual_state)
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle += noise[2]
        self.pole_velocity += noise[3]
        super().performAction(action)




if __name__ == '__main__':
    cp = CartPoleGaussianNoisyObs()
    cp.setState([0, 0, 0.1, 0])
    print(cp.getState())

    cp = CartPoleGaussianNoisyObsDyn()
    cp2 = CartPoleGaussianNoisyDyn()
    cp.setState([0, 0, 0.1, 0])
    cp2.setState([0, 0, 0.1, 0])

    print(cp.getState(), cp2.getState())
    cp.performAction()
    cp2.performAction()

    print(cp.getState(), cp2.getState())







