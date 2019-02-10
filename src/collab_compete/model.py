import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, layer_in_out, seed):
        super(Actor, self).__init__()
        print('[Actor] Initializing the Actor network ..... ')
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0], layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], action_size)
        self.bn1 = nn.BatchNorm1d(layer_in_out[0])
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        x = F.relu(self.fc1(states))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, layer_in_out, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): number of dimensions for input layer
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        input_shape = (state_size + action_size) * 2
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_shape, layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0], layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], 1)
        self.bn1 = nn.BatchNorm1d(layer_in_out[0])
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        # #print('Model Critic', states.shape, actions.shape)
        # print(states.shape, actions.shape)
        xs = torch.cat((states, actions), dim=1)
        # print('34343 ', xs.shape)
        # print(self.fc1(xs).shape)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

