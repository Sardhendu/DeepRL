
import numpy as np
import copy
import torch




class OUNoise:
    """Ornstein-Uhlenbeck process.
    
    OUNoise is helpful for continuous actions space where we cannot take random actions since the action
    space in continuous interim can be infinite
    
    # from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    """

    def __init__(self, size, seed, scale=0.1, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        print('[INIT] Initializing Ornstein-Uhlenbeck Noise for policy exploration ... ... ...')
        np.random.seed(seed)
        self.scale = scale
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()



class EpsilonGreedy:
    pass


    
# debug()

# 32208 + 149411 + 7877 + 511807 + 391050 + 27224 + 181
#
# 1020000 - (29*1632 + 209804 + 8780 + 8950 + 8950 + 172776 + 200000 + 200000)


#
# class OUNoise:
#
#     def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
#         self.action_dimension = action_dimension
#         self.scale = scale
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.state = np.ones(self.action_dimension) * self.mu
#         self.reset()
#
#     def reset(self):
#         self.state = np.ones(self.action_dimension) * self.mu
#
#     def noise(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
#         self.state = x + dx
#         return torch.tensor(self.state * self.scale).float()