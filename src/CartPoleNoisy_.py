from CartPole_ import CartPole
from numpy.random import (
    default_rng,
)

class Noise():
    def __init__(self, type='Gaussian') -> None:
        self.type = type
        self.rng = default_rng(10)
    
    def sample(self, size, *args, **kwargs):
        if self.type == 'Gaussian':
            loc, scale = kwargs['loc'], kwargs['scale']
            return self.rng.normal(loc, scale, size)
        else:
            pass


class CartPoleNoisy(CartPole):
    def __init__(self, visual=False):
        super().__init__(visual)
        self.noise_gen = Noise('Gaussian')   

    def getStateNoisy(self, *args, **kwargs):
        true_state = super().getState()
        noise = self.noise_gen.sample(4, **kwargs)
        return true_state + noise
    
    def performActionNoisy(self, action=0, *args, **kwargs):
        super().performAction(action)
        noise = self.noise_gen.sample(4, **kwargs)
        self.cart_location += noise[0]
        self.cart_velocity += noise[1]
        self.pole_angle += noise[2]
        self.pole_velocity += noise[3]


if __name__ == '__main__':
    cp = CartPoleNoisy()
    cp.setState([0, 0, 0.1, 0])
    print(cp.cart_location, cp.cart_velocity, cp.pole_angle, cp.pole_velocity)
    print(cp.getStateNoisy(loc=0, scale=1))
    print(cp.getState())
