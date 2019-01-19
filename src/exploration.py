
import numpy as np
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        print('[INIT] Initializing Ornstein-Uhlenbeck Noise for policy exploration ... ... ...')
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state



class EpsilonGreedy:
    pass



# 32208 + 149411 + 7877 + 511807 + 391050 + 27224 + 181
#
# 1020000 - (29*1632 + 209804 + 8780 + 8950 + 8950 + 172776 + 200000 + 200000)