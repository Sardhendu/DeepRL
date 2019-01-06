
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        :param state_size:      Dimension of state
        :param action_size:     Number of action
        """
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        """Forward propagation
        
        :param x:
        :return:
        
        Since we are dealing with continuous space we have to use a tanh nonlinearity. The tanh nonlinearity squashes the output between -1 and 1. We would also go away with a
        sigmoid depending on our action value limits.
        
        If the limit says -n to n, then we just multiply n to the tanh output.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
    
class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """THe DDPG Critic would output a single value for the input <state, action>
        
        :param state_size:
        :param action_size:
        :param seed:
        """
        super(DDPGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64 + action_size, 64)
        self.fc3 = nn.Linear(64, 1)
        
    
    def forward(self, state, action_dist):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action_dist), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Optimize:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def adam(self, params):
        return optim.Adam(params, lr=self.learning_rate)
    
    def rmsprop(self, params):
        return optim.RMSprop(params, lr=self.learning_rate)  # default smoothing=0.99, eps = 1e-08, momentum=0



def debug():
    state_torch = torch.from_numpy(np.random.random((1,33))).float().to(device)
    
    actor_local_network = DDPGActor(state_size=33, action_size=4)
    critic_local_network = DDPGCritic(state_size=33, action_size=4)
    a = actor_local_network.forward(state_torch)
    b = critic_local_network.forward(state_torch, a)
    
    
    print(a.shape, a)
    print(b)