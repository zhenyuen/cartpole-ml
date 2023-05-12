import CartPole as cp


class Simulation:
    def __init__(self):
        self.system = cp.CartPole(True)

    def run(self):
        self.system.performAction()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()

