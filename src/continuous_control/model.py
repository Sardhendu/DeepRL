
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
#
# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.bn1 = nn.BatchNorm1d(fc1_units)
#         # self.drop1 = nn.Dropout(p=0.7)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.bn2 = nn.BatchNorm1d(fc2_units)
#         # self.drop2 = nn.Dropout(p=0.5)
#         # self.bn2 = nn.BatchNorm1d(fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = F.relu(self.bn1(self.fc1(state)))
#         # x = self.drop1(x)
#         x = F.relu(self.bn2(self.fc2(x)))
#         # x = self.drop2(x)
#         return F.tanh(self.fc3(x))
#
#
# class Critic(nn.Module):
#     """Critic (Value) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=512):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fcs1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         # self.drop1 = nn.Dropout(p=0.7)
#         self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
#         self.drop2 = nn.Dropout(p=0.5)
#         self.fc3 = nn.Linear(fc2_units, 1)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, state, action):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         xs = F.relu(self.fc1(state))
#         # xs = self.drop1(xs)
#         x = torch.cat((xs, action), dim=1)
#         x = F.relu(self.fc2(x))
#         x = self.drop2(x)
#         return self.fc3(x)


class Actor(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, final_gate=F.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
            final_gate (function): final activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.final_gate = final_gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size,) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(states)
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.final_gate(self.output(x))


class Critic(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, dropout=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size,) + hidden_units
        self.layers = nn.ModuleList()
        count = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if count == 1:
                self.layers.append(nn.Linear(dim_in + action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            count += 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.normalizer(states)
        xs = self.gate(self.layers[0](xs))
        print(xs.shape)
        x = torch.cat((xs, actions), dim=1)
        for i in range(1, len(self.layers)):
            x = self.gate(self.layers[i](x))
        x = self.dropout(x)
        return self.output(x)

